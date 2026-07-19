"""Source-bound reports for the core code-review evaluation system.

``summary.json`` is the authoritative report projection.  Markdown is a pure,
deterministic rendering of an already-built report and never re-runs a scorer,
an evaluator, an Agent, or a model.  Prompt/model/provider/runtime metadata is
kept as provenance only; it is not converted into another product score.

The two public report artifacts are sealed.  They can only be created by
``ReportBuilder`` or hydrated by replaying that builder against the original
Run, Case, Submission, evaluator-result, and score sources.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .artifacts import RunManifest, StageReceipt, TrialManifest
from .cases import CaseDimension, SuiteCase
from .config import (
    EvalRunConfig,
    EvaluatorExecutionConfig,
    derive_case_path_id,
    derive_evaluation_id,
    derive_trial_seed,
    validate_evaluation_id,
    validate_path_segment,
    validate_safe_json,
    validate_trial_id_shape,
)
from .intent_evaluator import (
    IntentEvaluationResult,
    IntentEvaluationStatus,
)
from .metrics import (
    AggregateScore,
    CaseScore,
    CoreMetric,
    DEFAULT_METRICS_POLICY,
    MetricSourceStatus,
    MetricsAggregator,
    MetricsPolicy,
    ScoreCompatibilityKey,
    ScoreRef,
    TrialScore,
    TrialScorer,
)
from .models import (
    EvalCase,
    EvalSubmission,
    FindingSeverity,
    IssueJudgement,
    SchemaError,
    SubmissionStatus,
    TraceRef,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
    _strict_json_loads,
)
from .review_evaluator import (
    FindingDisposition,
    ReviewEvaluationResult,
    ReviewEvaluationStatus,
)


RUN_REPORT_SUMMARY_SCHEMA_VERSION = "eval_run_report_summary_v1"
TRIAL_INSPECTION_SCHEMA_VERSION = "eval_trial_inspection_v1"
REPORT_REVISION = "core-code-review-report-v1"
REDACTED_ARTIFACT_PROJECTION_VERSION = "eval_redacted_artifact_projection_v1"
TRACE_REF_SOURCE_SCHEMA_VERSION = "eval_trace_ref_v1"

_SUMMARY_SEAL_TOKEN = object()
_INSPECTION_SEAL_TOKEN = object()

MAX_REPORT_BYTES = 256 * 1024 * 1024
MAX_INSPECTION_BYTES = 256 * 1024 * 1024
MAX_REPORT_CASES = 65_536
MAX_REPORT_TRIALS = 65_536
MAX_TIMELINE_RECEIPTS = 4_096
MAX_TRACE_FILES = 4_096
MAX_GROUP_DIMENSIONS = 64

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class ReportError(ValueError):
    """A report source, persisted report, or rendering input is invalid."""


def _error(message: str) -> ReportError:
    return ReportError(message)


def _wire(value: Any, context: str = "value") -> Any:
    """Convert one bounded project value to its JSON projection."""

    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _wire(value.to_dict(), context)
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise _error(f"{context} contains a non-string object key")
            result[key] = _wire(item, f"{context}.{key}")
        return result
    if type(value) in (tuple, list):
        return [_wire(item, f"{context} item") for item in value]
    raise _error(f"{context} contains a non-JSON value")


def _object(value: Any, context: str) -> Dict[str, Any]:
    result = _wire(value, context)
    if type(result) is not dict:
        raise _error(f"{context} must be an object")
    return result


def _canonical_payload(
    value: Any,
    context: str,
    *,
    maximum: int = MAX_REPORT_BYTES,
) -> bytes:
    try:
        result = canonical_json_bytes(_wire(value, context))
    except (SchemaError, TypeError, ValueError) as exc:
        raise _error(f"{context} is not canonical JSON: {exc}") from exc
    if len(result) > maximum:
        raise _error(f"{context} exceeds its canonical byte budget")
    return result


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json(_wire(value)))


def _digest(value: Any, context: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise _error(f"{context} must be a canonical SHA-256 digest")
    return value


def _object_digest(value: Any, context: str) -> str:
    if hasattr(value, "digest") and callable(value.digest):
        return _digest(value.digest(), f"{context} digest")
    return canonical_sha256(_wire(value, context))


def _status(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.value if isinstance(value, Enum) else str(value)


def _sequence(value: Any, context: str, maximum: int) -> Tuple[Any, ...]:
    if isinstance(value, Mapping):
        items = tuple(value.values())
    elif type(value) in (tuple, list):
        items = tuple(value)
    else:
        raise _error(f"{context} must be a bounded collection")
    if len(items) > maximum:
        raise _error(f"{context} exceeds its item limit")
    return items


def _breakdown(values: Iterable[str]) -> list[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return [
        {"key": key, "count": counts[key]}
        for key in sorted(counts)
    ]


def _score_ref(score: Any, task_id: str, trial_id: Optional[str]) -> Dict[str, Any]:
    return ScoreRef(
        score_id=score.score_id,
        score_digest=score.digest(),
        task_id=task_id,
        trial_id=trial_id,
    ).to_dict()


def _canonical_receipts(values: Sequence[Any]) -> Tuple[Any, ...]:
    receipts = tuple(values)
    if len(receipts) > MAX_TIMELINE_RECEIPTS:
        raise _error("Trial timeline exceeds its receipt limit")
    keyed = []
    stage_order = {
        "start": 0,
        "incomplete": 1,
        "prepare": 2,
        "agent": 3,
        "evaluator": 4,
    }
    for item in receipts:
        payload = _object(item, "Trial timeline receipt")
        keyed.append(
            (
                (
                    stage_order.get(str(payload.get("stage")), 99),
                    payload.get("attempt") or 0,
                    payload.get("evaluation_id") or "",
                    payload.get("receipt_id") or "",
                    canonical_json(payload),
                ),
                item,
            )
        )
    if len({key for key, _item in keyed}) != len(keyed):
        raise _error("Trial timeline contains duplicate receipts")
    return tuple(item for _key, item in sorted(keyed, key=lambda entry: entry[0]))


def _trace_capture_projection(
    value: Any,
    *,
    maximum_total_bytes: Optional[int] = None,
    maximum_file_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Project trace metadata while making raw trace content unrepresentable."""

    if value is None:
        return {
            "captured": False,
            "reason": "metadata_not_provided",
            "total_bytes": None,
            "files": [],
        }
    raw = _object(value, "trace capture metadata")
    captured = raw.get("captured")
    if type(captured) is not bool:
        captured = bool(raw.get("files"))
    reason = raw.get("reason")
    if reason is not None and type(reason) is not str:
        reason = str(reason)
    total_bytes = raw.get("total_bytes")
    if total_bytes is not None and (type(total_bytes) is not int or total_bytes < 0):
        raise _error("trace capture total_bytes must be non-negative or null")
    raw_files = raw.get("files", [])
    if type(raw_files) not in (tuple, list):
        raise _error("trace capture files must be an array")
    if len(raw_files) > MAX_TRACE_FILES:
        raise _error("trace capture files exceed their item limit")
    files = []
    file_total = 0
    for raw_file in raw_files:
        item = _object(raw_file, "trace capture file")
        path = item.get("path")
        if type(path) is not str:
            raise _error("trace capture file.path must be a string")
        normalized = path.replace("\\", "/")
        if (
            normalized.startswith("/")
            or re.match(r"^[A-Za-z]:", normalized)
            or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", normalized)
            or any(part == ".." for part in normalized.split("/"))
        ):
            normalized = "<absolute-path-redacted>"
        size_bytes = item.get("size_bytes")
        if type(size_bytes) is not int or size_bytes < 0:
            raise _error("trace capture file.size_bytes must be non-negative")
        if maximum_file_bytes is not None and size_bytes > maximum_file_bytes:
            raise _error("trace capture file exceeds the Run trace budget")
        file_total += size_bytes
        sha256 = item.get("sha256")
        if sha256 is not None:
            _digest(sha256, "trace capture file.sha256")
        content_truncated = item.get("content_truncated", False)
        if type(content_truncated) is not bool:
            raise _error("trace capture file.content_truncated must be bool")
        files.append(
            {
                "path": normalized,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "content_truncated": content_truncated,
            }
        )
    if files and total_bytes is None:
        raise _error("trace capture with files must declare total_bytes")
    if total_bytes is not None and files and total_bytes != file_total:
        raise _error("trace capture total_bytes differs from file size coverage")
    if maximum_total_bytes is not None and total_bytes is not None:
        if total_bytes > maximum_total_bytes:
            raise _error("trace capture exceeds the Run trace budget")
    if captured is False and files:
        raise _error("uncaptured trace metadata cannot contain files")
    files.sort(key=lambda item: (item["path"], item["sha256"] or ""))
    return {
        "captured": captured,
        "reason": reason,
        "total_bytes": total_bytes,
        "files": files,
    }


def _safe_trace_ref_projection(value: Any) -> Dict[str, Any]:
    """Publish a source-bound redacted artifact, never a canonical TraceRef."""

    raw = _object(value, "trace reference")
    try:
        trace_ref = TraceRef.from_dict(raw)
    except (SchemaError, TypeError, ValueError) as exc:
        raise _error(f"trace reference is invalid: {exc}") from exc
    source_digest = trace_ref.digest()
    opaque_id = "trace-ref-" + source_digest
    projection = _redacted_artifact_projection(
        artifact_kind="trace_ref",
        source_payload={
            "schema_version": TRACE_REF_SOURCE_SCHEMA_VERSION,
            **trace_ref.to_dict(),
        },
        source_digest=source_digest,
        source_id=opaque_id,
    )
    projection["redactions"] = ["trace_ref:value"]
    projection["payload"] = {
        "type": "opaque_id",
        "value": opaque_id,
    }
    return projection


def _redact_sensitive_structure(value: Any, *, field_name: Optional[str] = None) -> Any:
    if type(value) is dict:
        return {
            key: _redact_sensitive_structure(child, field_name=key)
            for key, child in value.items()
        }
    if type(value) is list:
        return [
            _redact_sensitive_structure(child, field_name=field_name)
            for child in value
        ]
    if type(value) is not str:
        return value
    if field_name == "url":
        return "url-ref-" + canonical_sha256({"url": value})
    if field_name in {"path", "relative_path", "local_path"}:
        normalized = value.replace("\\", "/")
        if (
            normalized.startswith("/")
            or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", normalized)
            or any(part == ".." for part in normalized.split("/"))
        ):
            return "path-ref-" + canonical_sha256({"path": value})
    return value


def _redacted_artifact_projection(
    *,
    artifact_kind: str,
    source_payload: Mapping[str, Any],
    source_digest: str,
    source_id: str,
    remove_identity_fields: Sequence[str] = (),
) -> Dict[str, Any]:
    source = _json_copy(source_payload)
    source_schema_version = source.pop("schema_version")
    for field in remove_identity_fields:
        source.pop(field, None)
    payload = _redact_sensitive_structure(source)
    return {
        "schema_version": REDACTED_ARTIFACT_PROJECTION_VERSION,
        "artifact_kind": artifact_kind,
        "source_schema_version": source_schema_version,
        "source_id": source_id,
        "source_digest": source_digest,
        "redactions": ["structured_paths_and_urls"],
        "payload": payload,
    }


def _safe_submission_projection(value: EvalSubmission) -> Dict[str, Any]:
    source = value.to_dict()
    if source.get("trace_ref") is not None:
        source["trace_ref"] = _safe_trace_ref_projection(source["trace_ref"])
    projection = _redacted_artifact_projection(
        artifact_kind="eval_submission",
        source_payload=source,
        source_digest=value.digest(),
        source_id=value.trial_id,
    )
    if value.trace_ref is not None:
        projection["redactions"].append("trace_ref:opaque")
    return projection


def _safe_score_projection(value: TrialScore) -> Dict[str, Any]:
    source = value.to_dict()
    if source.get("trace_ref") is not None:
        source["trace_ref"] = _safe_trace_ref_projection(source["trace_ref"])
    projection = _redacted_artifact_projection(
        artifact_kind="trial_score",
        source_payload=source,
        source_digest=value.digest(),
        source_id=value.score_id,
        remove_identity_fields=("score_id",),
    )
    if value.trace_ref is not None:
        projection["redactions"].append("trace_ref:opaque")
    return projection


def _safe_eval_input_projection(value: Any) -> Dict[str, Any]:
    return _redacted_artifact_projection(
        artifact_kind="eval_input",
        source_payload=value.to_dict(),
        source_digest=value.digest(),
        source_id=value.task_id,
    )


def _safe_intent_evaluation_projection(value: IntentEvaluationResult) -> Dict[str, Any]:
    digest = value.digest()
    return _redacted_artifact_projection(
        artifact_kind="intent_evaluation",
        source_payload=value.to_dict(),
        source_digest=digest,
        source_id="intent-evaluation-" + digest,
    )


def _safe_review_evaluation_projection(value: ReviewEvaluationResult) -> Dict[str, Any]:
    digest = value.digest()
    return _redacted_artifact_projection(
        artifact_kind="review_evaluation",
        source_payload=value.to_dict(),
        source_digest=digest,
        source_id="review-evaluation-" + digest,
    )


def _timeline_projection(values: Sequence[Any]) -> list[Dict[str, Any]]:
    result = []
    for value in _canonical_receipts(values):
        raw = _object(value, "Trial timeline receipt")
        # StageReceipt is already a metadata-only artifact.  Keeping a fixed
        # allow-list prevents a future receipt extension from leaking content.
        allowed = (
            "schema_version",
            "receipt_id",
            "run_id",
            "task_id",
            "trial_id",
            "stage",
            "config_digest",
            "attempt",
            "evaluation_id",
            "evaluation_revision",
            "artifacts",
            "terminal_status",
            "failure_code",
        )
        projected = {name: raw[name] for name in allowed if name in raw}
        artifacts = projected.get("artifacts")
        if artifacts is not None:
            if type(artifacts) is not list:
                raise _error("timeline receipt artifacts must be an array")
            projected["artifacts"] = sorted(
                artifacts,
                key=lambda item: canonical_json(item),
            )
        result.append(projected)
    return result


@dataclass(frozen=True)
class TrialEvaluationSource:
    """The real source bundle for one planned Trial.

    ``trial_id`` and ``trial_index`` may be omitted when a Submission or
    TrialScore supplies them.  A source containing neither may still represent
    an explicit planned/nonterminal slot when both identifiers are supplied.
    """

    eval_case: EvalCase
    submission: Optional[EvalSubmission] = None
    intent_result: Optional[IntentEvaluationResult] = None
    review_result: Optional[ReviewEvaluationResult] = None
    trial_score: Optional[TrialScore] = None
    trial_index: Optional[int] = None
    trial_id: Optional[str] = None
    trial_manifest: Optional[Any] = None
    timeline: Tuple[Any, ...] = ()
    trace_capture: Optional[Any] = None

    def __post_init__(self) -> None:
        if type(self.eval_case) is not EvalCase:
            raise _error("TrialEvaluationSource requires an EvalCase")
        if self.submission is not None and type(self.submission) is not EvalSubmission:
            raise _error("TrialEvaluationSource submission is invalid")
        if self.intent_result is not None and type(self.intent_result) is not IntentEvaluationResult:
            raise _error("TrialEvaluationSource Intent result is invalid")
        if self.review_result is not None and type(self.review_result) is not ReviewEvaluationResult:
            raise _error("TrialEvaluationSource Review result is invalid")
        if self.trial_score is not None and type(self.trial_score) is not TrialScore:
            raise _error("TrialEvaluationSource Trial score is invalid")
        if self.trial_index is not None and (
            type(self.trial_index) is not int or self.trial_index < 1
        ):
            raise _error("TrialEvaluationSource trial_index must be positive")
        if self.trial_id is not None:
            try:
                validate_trial_id_shape(self.trial_id)
            except (SchemaError, ValueError) as exc:
                raise _error(str(exc)) from exc

        submission = self.submission
        case = self.eval_case
        if submission is not None:
            if submission.task_id != case.task_id:
                raise _error("Submission task_id differs from EvalCase")
            case.validate_submission(submission)
        elif any(
            value is not None
            for value in (self.intent_result, self.review_result, self.trial_score)
        ):
            raise _error("evaluator results and scores require their real Submission")

        if self.intent_result is not None:
            if submission is None or submission.intent is None:
                raise _error("Intent result has no matching Submission Intent")
            expected_intent_digest = canonical_sha256(submission.intent.to_dict())
            if (
                self.intent_result.submission_intent_digest != expected_intent_digest
                or self.intent_result.intent_truth_digest != case.intent_truth.digest()
                or self.intent_result.clarification_script_digest
                != case.clarification_script.digest()
            ):
                raise _error("Intent result is not bound to Submission/Case sources")
        elif submission is not None and submission.intent is None:
            pass

        if self.review_result is not None:
            if submission is None or submission.review is None:
                raise _error("Review result has no matching Submission Review")
            expected_review_digest = canonical_sha256(submission.review.to_dict())
            expected_evidence_digest = canonical_sha256(
                [item.to_dict() for item in submission.evidence]
            )
            if (
                self.review_result.submission_digest != submission.digest()
                or self.review_result.submission_review_digest
                != expected_review_digest
                or self.review_result.submission_evidence_digest
                != expected_evidence_digest
                or self.review_result.eval_input_digest != case.eval_input().digest()
                or self.review_result.review_truth_digest != case.review_truth.digest()
                or self.review_result.truth_completeness
                is not case.review_truth.completeness
                or self.review_result.novel_finding_policy
                is not case.review_truth.novel_finding_policy
            ):
                raise _error("Review result is not bound to Submission/Case sources")

        derived_trial_ids = []
        if submission is not None:
            derived_trial_ids.append(submission.trial_id)
        if self.trial_score is not None:
            score = self.trial_score
            derived_trial_ids.append(score.trial_id)
            if (
                score.task_id != case.task_id
                or score.case_version != case.case_version
                or score.canonical_case_digest != case.digest()
                or score.eval_input_digest != case.eval_input().digest()
                or submission is None
                or score.submission_digest != submission.digest()
                or score.submission_status is not submission.status
                or score.usage != submission.usage
                or score.trace_ref != submission.trace_ref
            ):
                raise _error("Trial score is not bound to Submission/Case sources")
            if self.trial_index is not None and score.trial_index != self.trial_index:
                raise _error("Trial score and source trial_index differ")
            if score.intent_binding is None:
                if (
                    self.intent_result is not None
                    and self.intent_result.status
                    is not IntentEvaluationStatus.PENDING_JUDGE
                ):
                    raise _error("Trial score omits its supplied Intent result")
            elif (
                self.intent_result is None
                or self.intent_result.status is IntentEvaluationStatus.PENDING_JUDGE
                or score.intent_binding.result_digest != self.intent_result.digest()
            ):
                raise _error("Trial score Intent binding differs from source result")
            if score.review_binding is None:
                if (
                    self.review_result is not None
                    and self.review_result.status
                    is not ReviewEvaluationStatus.PENDING_JUDGE
                ):
                    raise _error("Trial score omits its supplied Review result")
            elif (
                self.review_result is None
                or self.review_result.status is ReviewEvaluationStatus.PENDING_JUDGE
                or score.review_binding.result_digest != self.review_result.digest()
            ):
                raise _error("Trial score Review binding differs from source result")

        if len(set(derived_trial_ids)) > 1:
            raise _error("Trial source objects refer to different trial IDs")
        derived_trial_id = derived_trial_ids[0] if derived_trial_ids else None
        if self.trial_id is not None and derived_trial_id is not None:
            if self.trial_id != derived_trial_id:
                raise _error("explicit trial_id differs from source artifacts")
        elif self.trial_id is None and derived_trial_id is not None:
            object.__setattr__(self, "trial_id", derived_trial_id)

        canonical_timeline = []
        for value in self.timeline:
            if type(value) is StageReceipt:
                canonical_timeline.append(value)
                continue
            try:
                canonical_timeline.append(StageReceipt.from_dict(_object(value, "Trial timeline receipt")))
            except (SchemaError, TypeError, ValueError) as exc:
                raise _error(f"Trial timeline receipt is invalid: {exc}") from exc
        object.__setattr__(self, "timeline", _canonical_receipts(canonical_timeline))
        if self.trial_manifest is not None:
            if type(self.trial_manifest) is not TrialManifest:
                try:
                    object.__setattr__(
                        self,
                        "trial_manifest",
                        TrialManifest.from_dict(
                            _object(self.trial_manifest, "Trial manifest")
                        ),
                    )
                except (SchemaError, TypeError, ValueError) as exc:
                    raise _error(f"Trial manifest is invalid: {exc}") from exc
            _canonical_payload(self.trial_manifest, "Trial manifest", maximum=MAX_INSPECTION_BYTES)
        if self.trace_capture is not None:
            _trace_capture_projection(self.trace_capture)

    def to_dict(self) -> Dict[str, Any]:
        """Return a metadata-only source binding, never the private truth body."""

        return {
            "task_id": self.eval_case.task_id,
            "case_version": self.eval_case.case_version,
            "trial_id": self.trial_id,
            "trial_index": self.trial_index,
            "canonical_case_digest": self.eval_case.digest(),
            "eval_input_digest": self.eval_case.eval_input().digest(),
            "submission_digest": (
                None if self.submission is None else self.submission.digest()
            ),
            "intent_result_digest": (
                None if self.intent_result is None else self.intent_result.digest()
            ),
            "review_result_digest": (
                None if self.review_result is None else self.review_result.digest()
            ),
            "trial_score_digest": (
                None if self.trial_score is None else self.trial_score.digest()
            ),
        }


@dataclass(frozen=True)
class _ResolvedTrial:
    source: TrialEvaluationSource
    eval_case: EvalCase
    suite_case: SuiteCase
    trial_index: int
    trial_id: str
    score: Optional[TrialScore]


@dataclass(frozen=True, init=False)
class RunReportSummary:
    """A sealed, canonical ``summary.json`` projection."""

    _canonical_json: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "RunReportSummary must be created by ReportBuilder or source-bound hydration"
        )

    @classmethod
    def _seal(
        cls,
        payload: Mapping[str, Any],
        *,
        _token: object = None,
    ) -> "RunReportSummary":
        if _token is not _SUMMARY_SEAL_TOKEN:
            raise TypeError("RunReportSummary can only be sealed by ReportBuilder")
        result = object.__new__(cls)
        object.__setattr__(result, "_canonical_json", canonical_json(_wire(payload)))
        result.__post_init__()
        return result

    def __post_init__(self) -> None:
        payload = json.loads(self._canonical_json)
        expected_fields = {
            "schema_version",
            "report_revision",
            "summary_id",
            "source_bindings",
            "identity",
            "coverage",
            "partitions",
            "cases",
            "diagnostics",
        }
        if set(payload) != expected_fields:
            raise _error("Run report summary has an invalid top-level schema")
        if payload["schema_version"] != RUN_REPORT_SUMMARY_SCHEMA_VERSION:
            raise _error("Run report summary has an unsupported schema version")
        if payload["report_revision"] != REPORT_REVISION:
            raise _error("Run report summary has an unsupported report revision")
        identity = dict(payload)
        summary_id = identity.pop("summary_id")
        if summary_id != stable_id("run-report-summary-v1", identity):
            raise _error("Run report summary ID is not canonical")
        try:
            validate_safe_json(payload, "Run report summary")
        except (SchemaError, ValueError) as exc:
            raise _error(str(exc)) from exc
        _canonical_payload(payload, "Run report summary")

    def _value(self, name: str) -> Any:
        return json.loads(self._canonical_json)[name]

    @property
    def schema_version(self) -> str:
        return self._value("schema_version")

    @property
    def report_revision(self) -> str:
        return self._value("report_revision")

    @property
    def summary_id(self) -> str:
        return self._value("summary_id")

    @property
    def source_bindings(self) -> Dict[str, Any]:
        return self._value("source_bindings")

    @property
    def identity(self) -> Dict[str, Any]:
        return self._value("identity")

    @property
    def coverage(self) -> Dict[str, Any]:
        return self._value("coverage")

    @property
    def partitions(self) -> list[Dict[str, Any]]:
        return self._value("partitions")

    @property
    def cases(self) -> list[Dict[str, Any]]:
        return self._value("cases")

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return self._value("diagnostics")

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(self._canonical_json)

    def to_json(self) -> str:
        return self._canonical_json

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        run_config: EvalRunConfig,
        evaluator_execution: EvaluatorExecutionConfig,
        evaluation_revision: str,
        eval_cases: Optional[Sequence[EvalCase]] = None,
        trial_sources: Optional[Sequence[TrialEvaluationSource]] = None,
        cases: Optional[Sequence[EvalCase]] = None,
        sources: Optional[Sequence[TrialEvaluationSource]] = None,
        builder: Optional["ReportBuilder"] = None,
        group_dimension_names: Optional[Sequence[str]] = None,
        run_manifest: Optional[Any] = None,
    ) -> "RunReportSummary":
        if type(value) is not dict:
            raise _error("Run report summary payload must be an object")
        if builder is not None and type(builder) is not ReportBuilder:
            raise _error("Run report summary hydration requires ReportBuilder")
        report_builder = ReportBuilder() if builder is None else builder
        replayed = ReportBuilder.build_summary(
            report_builder,
            run_config=run_config,
            evaluator_execution=evaluator_execution,
            evaluation_revision=evaluation_revision,
            eval_cases=eval_cases,
            trial_sources=trial_sources,
            cases=cases,
            sources=sources,
            group_dimension_names=group_dimension_names,
            run_manifest=run_manifest,
        )
        if _canonical_payload(value, "Run report summary payload") != canonical_json_bytes(
            replayed.to_dict()
        ):
            raise _error("persisted Run report summary differs from source-bound replay")
        return replayed

    @classmethod
    def from_json(cls, data: Any, **sources: Any) -> "RunReportSummary":
        try:
            payload = _strict_json_loads(
                data,
                MAX_REPORT_BYTES,
                "Run report summary JSON",
            )
        except (SchemaError, TypeError, ValueError) as exc:
            raise _error(str(exc)) from exc
        return cls.from_dict(payload, **sources)

    serialize = to_dict
    hydrate = from_dict


@dataclass(frozen=True, init=False)
class TrialInspection:
    """A sealed, metadata-safe inspection projection for one Trial."""

    _canonical_json: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "TrialInspection must be created by ReportBuilder or source-bound hydration"
        )

    @classmethod
    def _seal(
        cls,
        payload: Mapping[str, Any],
        *,
        _token: object = None,
    ) -> "TrialInspection":
        if _token is not _INSPECTION_SEAL_TOKEN:
            raise TypeError("TrialInspection can only be sealed by ReportBuilder")
        result = object.__new__(cls)
        object.__setattr__(result, "_canonical_json", canonical_json(_wire(payload)))
        result.__post_init__()
        return result

    def __post_init__(self) -> None:
        payload = json.loads(self._canonical_json)
        expected_fields = {
            "schema_version",
            "report_revision",
            "inspection_id",
            "source_bindings",
            "trial_manifest",
            "timeline",
            "input",
            "submission",
            "score",
            "intent_evaluation",
            "review_evaluation",
            "judge_artifact_refs",
            "clarification_match_receipts",
            "evidence_diagnostics",
            "trace",
        }
        if set(payload) != expected_fields:
            raise _error("Trial inspection has an invalid top-level schema")
        if payload["schema_version"] != TRIAL_INSPECTION_SCHEMA_VERSION:
            raise _error("Trial inspection has an unsupported schema version")
        if payload["report_revision"] != REPORT_REVISION:
            raise _error("Trial inspection has an unsupported report revision")
        identity = dict(payload)
        inspection_id = identity.pop("inspection_id")
        if inspection_id != stable_id("trial-inspection-v1", identity):
            raise _error("Trial inspection ID is not canonical")
        try:
            validate_safe_json(payload, "Trial inspection")
        except (SchemaError, ValueError) as exc:
            raise _error(str(exc)) from exc
        _canonical_payload(
            payload,
            "Trial inspection",
            maximum=MAX_INSPECTION_BYTES,
        )

    def _value(self, name: str) -> Any:
        return json.loads(self._canonical_json)[name]

    @property
    def schema_version(self) -> str:
        return self._value("schema_version")

    @property
    def report_revision(self) -> str:
        return self._value("report_revision")

    @property
    def inspection_id(self) -> str:
        return self._value("inspection_id")

    @property
    def source_bindings(self) -> Dict[str, Any]:
        return self._value("source_bindings")

    @property
    def timeline(self) -> list[Dict[str, Any]]:
        return self._value("timeline")

    @property
    def input(self) -> Dict[str, Any]:
        return self._value("input")

    @property
    def submission(self) -> Optional[Dict[str, Any]]:
        return self._value("submission")

    @property
    def score(self) -> Optional[Dict[str, Any]]:
        return self._value("score")

    @property
    def intent_evaluation(self) -> Optional[Dict[str, Any]]:
        return self._value("intent_evaluation")

    @property
    def review_evaluation(self) -> Optional[Dict[str, Any]]:
        return self._value("review_evaluation")

    @property
    def judge_artifact_refs(self) -> Dict[str, Any]:
        return self._value("judge_artifact_refs")

    @property
    def clarification_match_receipts(self) -> Optional[Dict[str, Any]]:
        return self._value("clarification_match_receipts")

    @property
    def evidence_diagnostics(self) -> list[Dict[str, Any]]:
        return self._value("evidence_diagnostics")

    @property
    def trace(self) -> Dict[str, Any]:
        return self._value("trace")

    @property
    def trial_manifest(self) -> Optional[Dict[str, Any]]:
        return self._value("trial_manifest")

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(self._canonical_json)

    def to_json(self) -> str:
        return self._canonical_json

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        run_config: EvalRunConfig,
        evaluator_execution: EvaluatorExecutionConfig,
        evaluation_revision: str,
        trial_source: Optional[TrialEvaluationSource] = None,
        source: Optional[TrialEvaluationSource] = None,
        builder: Optional["ReportBuilder"] = None,
        run_manifest: Optional[Any] = None,
    ) -> "TrialInspection":
        if type(value) is not dict:
            raise _error("Trial inspection payload must be an object")
        actual_source = trial_source if trial_source is not None else source
        if actual_source is None:
            raise _error("Trial inspection hydration requires TrialEvaluationSource")
        if builder is not None and type(builder) is not ReportBuilder:
            raise _error("Trial inspection hydration requires ReportBuilder")
        report_builder = ReportBuilder() if builder is None else builder
        replayed = ReportBuilder.build_inspection(
            report_builder,
            run_config=run_config,
            evaluator_execution=evaluator_execution,
            evaluation_revision=evaluation_revision,
            trial_source=actual_source,
            run_manifest=run_manifest,
        )
        if _canonical_payload(
            value,
            "Trial inspection payload",
            maximum=MAX_INSPECTION_BYTES,
        ) != canonical_json_bytes(replayed.to_dict()):
            raise _error("persisted Trial inspection differs from source-bound replay")
        return replayed

    @classmethod
    def from_json(cls, data: Any, **sources: Any) -> "TrialInspection":
        try:
            payload = _strict_json_loads(
                data,
                MAX_INSPECTION_BYTES,
                "Trial inspection JSON",
            )
        except (SchemaError, TypeError, ValueError) as exc:
            raise _error(str(exc)) from exc
        return cls.from_dict(payload, **sources)

    serialize = to_dict
    hydrate = from_dict


def _judge_reference(
    value: Any,
    *,
    phase: str,
    kind: str,
    parent_result_digest: str,
) -> Dict[str, Any]:
    raw = _object(value, f"{phase} Judge {kind}")
    result: Dict[str, Any] = {
        "phase": phase,
        "kind": kind,
        "request_id": raw.get("request_id"),
        "task": raw.get("task", phase),
        "request_digest": raw.get("request_digest"),
        "evaluator_execution_digest": raw.get("evaluator_execution_digest"),
        "judge_result_digest": raw.get("judge_result_digest"),
        "blind_request_id": raw.get("blind_request_id"),
        "parent_result_digest": parent_result_digest,
    }
    if result["request_digest"] is None:
        request = raw.get("request")
        if request is not None:
            result["request_digest"] = canonical_sha256(request)
    if kind == "decision":
        decision = raw.get("decision", raw)
        result["decision_digest"] = canonical_sha256(decision)
        if type(decision) is dict:
            result["reason_refs"] = list(decision.get("reason_refs", []))
    elif kind == "failure":
        failure = raw.get("failure")
        if type(failure) is dict:
            result["failure_code"] = failure.get("code")
        else:
            result["failure_code"] = raw.get("failure_code")
    elif kind == "ungraded":
        result["ungraded_reason"] = raw.get("ungraded_reason")
    return result


def _judge_artifact_refs(
    intent_result: Optional[IntentEvaluationResult],
    review_result: Optional[ReviewEvaluationResult],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for phase, evaluation in (
        ("intent", intent_result),
        ("review", review_result),
    ):
        phase_result: Dict[str, Any] = {
            "requests": [],
            "decisions": [],
            "failures": [],
            "ungraded": [],
        }
        if evaluation is not None:
            parent_digest = evaluation.digest()
            for attr, kind in (
                ("judge_requests", "request"),
                ("judge_decisions", "decision"),
                ("judge_failures", "failure"),
                ("judge_ungraded", "ungraded"),
            ):
                projected = [
                    _judge_reference(
                        item,
                        phase=phase,
                        kind=kind,
                        parent_result_digest=parent_digest,
                    )
                    for item in getattr(evaluation, attr)
                ]
                projected.sort(
                    key=lambda item: (
                        item.get("request_id") or "",
                        canonical_json(item),
                    )
                )
                phase_result[attr.removeprefix("judge_")] = projected
        result[phase] = phase_result
    return result


def _clarification_projection(
    result: Optional[IntentEvaluationResult],
) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    evaluation = result.clarification.to_dict()
    receipts = []
    for exchange in evaluation["exchanges"]:
        if exchange["receipt_digest"] is None:
            continue
        receipts.append(
            {
                "turn_index": exchange["turn_index"],
                "question_id": exchange["question_id"],
                "matched_answer_id": exchange["matched_answer_id"],
                "receipt_digest": exchange["receipt_digest"],
                "matcher_digest": exchange["matcher_digest"],
                "material": exchange["material"],
                "answer_consumed": exchange["answer_consumed"],
                "update_applied": exchange["update_applied"],
                "reason_codes": list(exchange["reason_codes"]),
            }
        )
    receipts.sort(key=lambda item: (item["turn_index"], item["question_id"]))
    return {
        "evaluation_digest": canonical_sha256(evaluation),
        "evaluation": evaluation,
        "receipt_refs": receipts,
    }


def _evidence_projection(
    result: Optional[ReviewEvaluationResult],
) -> list[Dict[str, Any]]:
    if result is None:
        return []
    projected = [
        _redact_sensitive_structure(item.to_dict())
        for item in result.evidence_integrity_results
    ]
    projected.sort(key=lambda item: item["finding_id"])
    return projected


def _semantic_unknown(value: Any) -> Optional[Dict[str, Any]]:
    raw = _object(value, "Judge decision")
    decision = raw.get("decision", raw)
    if type(decision) is not dict:
        return None
    unknown_fields = {
        name: decision[name]
        for name in ("relation", "factuality", "support", "judgement")
        if decision.get(name) == "unknown"
    }
    if not unknown_fields:
        return None
    return {
        "request_id": raw.get("request_id", decision.get("request_id")),
        "task": raw.get("task"),
        "unknown_fields": unknown_fields,
        "reason_refs": list(decision.get("reason_refs", [])),
        "decision_digest": canonical_sha256(decision),
        "evaluator_execution_digest": raw.get("evaluator_execution_digest"),
        "judge_result_digest": raw.get("judge_result_digest"),
    }


class ReportBuilder:
    """Build sealed Run summaries and per-Trial inspection projections."""

    __slots__ = ("metrics_policy", "scorer", "aggregator", "_sealed")

    def __init__(
        self,
        metrics_policy: MetricsPolicy = DEFAULT_METRICS_POLICY,
        *,
        scorer: Optional[TrialScorer] = None,
        aggregator: Optional[MetricsAggregator] = None,
    ) -> None:
        if type(metrics_policy) is not MetricsPolicy:
            raise _error("ReportBuilder requires MetricsPolicy")
        if scorer is None:
            scorer = TrialScorer(metrics_policy)
        if type(scorer) is not TrialScorer:
            raise _error("ReportBuilder scorer must be TrialScorer")
        try:
            scorer = TrialScorer._canonical_clone(scorer)
        except (SchemaError, TypeError, ValueError) as exc:
            raise _error(f"ReportBuilder scorer is invalid: {exc}") from exc
        if canonical_json_bytes(scorer.metrics_policy.to_dict()) != canonical_json_bytes(
            metrics_policy.to_dict()
        ):
            raise _error("ReportBuilder scorer and MetricsPolicy differ")
        if aggregator is None:
            aggregator = MetricsAggregator()
        if type(aggregator) is not MetricsAggregator:
            raise _error("ReportBuilder aggregator must be MetricsAggregator")
        object.__setattr__(self, "metrics_policy", scorer.metrics_policy)
        object.__setattr__(self, "scorer", scorer)
        object.__setattr__(self, "aggregator", MetricsAggregator())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ReportBuilder configuration is immutable")
        object.__setattr__(self, name, value)

    def _trusted_components(
        self,
    ) -> Tuple[MetricsPolicy, TrialScorer, MetricsAggregator]:
        """Snapshot canonical components instead of trusting mutable slots."""

        if type(self) is not ReportBuilder:
            raise _error("report construction requires an exact ReportBuilder")
        if type(self.metrics_policy) is not MetricsPolicy:
            raise _error("ReportBuilder MetricsPolicy was mutated")
        if type(self.scorer) is not TrialScorer:
            raise _error("ReportBuilder TrialScorer was mutated")
        if type(self.aggregator) is not MetricsAggregator:
            raise _error("ReportBuilder MetricsAggregator was mutated")
        try:
            scorer = TrialScorer._canonical_clone(self.scorer)
        except (SchemaError, TypeError, ValueError) as exc:
            raise _error(f"ReportBuilder TrialScorer is invalid: {exc}") from exc
        if canonical_json_bytes(self.metrics_policy.to_dict()) != canonical_json_bytes(
            scorer.metrics_policy.to_dict()
        ):
            raise _error("ReportBuilder scorer and MetricsPolicy differ")
        return scorer.metrics_policy, scorer, MetricsAggregator()

    @staticmethod
    def _validate_run_sources(
        run_config: EvalRunConfig,
        evaluator_execution: EvaluatorExecutionConfig,
        evaluation_revision: str,
    ) -> Tuple[str, str]:
        if type(run_config) is not EvalRunConfig:
            raise _error("ReportBuilder requires EvalRunConfig")
        if type(evaluator_execution) is not EvaluatorExecutionConfig:
            raise _error("ReportBuilder requires EvaluatorExecutionConfig")
        try:
            revision = validate_path_segment(
                evaluation_revision,
                "evaluation revision",
            )
        except (SchemaError, ValueError) as exc:
            raise _error(str(exc)) from exc
        # The immutable Run stores the evaluator snapshot that was current
        # when Agent execution was planned, but evaluator/Judge identity is
        # deliberately absent from the Run ID.  Re-evaluation may therefore
        # supply a different, fully versioned EvaluatorExecutionConfig.  Its
        # digest plus ``evaluation_revision`` creates a new evaluation ID and
        # every downstream result/receipt is source-bound to that execution.
        # Requiring equality with ``run_config.evaluator`` here would make the
        # documented "change Judge without rerunning the Agent" workflow
        # impossible.
        execution_digest = evaluator_execution.digest()
        evaluation_id = derive_evaluation_id(
            run_config.run_id,
            execution_digest,
            revision,
        )
        validate_evaluation_id(
            evaluation_id,
            run_config.run_id,
            execution_digest,
            revision,
        )
        return revision, evaluation_id

    @staticmethod
    def _normalize_cases(
        run_config: EvalRunConfig,
        eval_cases: Sequence[EvalCase],
    ) -> Tuple[EvalCase, ...]:
        values = _sequence(eval_cases, "EvalCase collection", MAX_REPORT_CASES)
        if not values or any(type(item) is not EvalCase for item in values):
            raise _error("EvalCase collection is empty or contains an invalid item")
        if len({item.task_id for item in values}) != len(values):
            raise _error("EvalCase collection contains duplicate task IDs")
        ordered = tuple(sorted(values, key=lambda item: item.task_id))
        expected_task_ids = {item.task_id for item in run_config.suite.cases}
        actual_task_ids = {item.task_id for item in ordered}
        if actual_task_ids != expected_task_ids:
            raise _error("EvalCase collection differs from the immutable Run Suite")
        for case in ordered:
            try:
                suite_case = run_config.suite.case(case.task_id)
            except (SchemaError, ValueError) as exc:
                raise _error(str(exc)) from exc
            if (
                case.case_version != suite_case.case_version
                or case.digest() != suite_case.canonical_case_digest
                or case.eval_input().digest() != suite_case.eval_input_digest
                or case.review_truth.completeness
                is not suite_case.truth_completeness
                or case.source.suite != run_config.suite.suite_id
            ):
                raise _error("EvalCase differs from its immutable Suite binding")
        return ordered

    @staticmethod
    def _find_trial_index(
        run_config: EvalRunConfig,
        task_id: str,
        trial_id: str,
    ) -> int:
        for index in range(1, run_config.trial_count + 1):
            if run_config.trial_id(task_id, index) == trial_id:
                return index
        raise _error("trial_id is not one of the immutable Run Trial slots")

    def _resolve_trial(
        self,
        *,
        run_config: EvalRunConfig,
        evaluator_execution: EvaluatorExecutionConfig,
        evaluation_revision: str,
        case_by_task: Mapping[str, EvalCase],
        source: TrialEvaluationSource,
        metrics_policy: MetricsPolicy,
        scorer: TrialScorer,
    ) -> _ResolvedTrial:
        if type(source) is not TrialEvaluationSource:
            raise _error("Trial source collection contains an invalid item")
        canonical_case = case_by_task.get(source.eval_case.task_id)
        if canonical_case is None or canonical_case.digest() != source.eval_case.digest():
            raise _error("Trial source EvalCase is outside the Run Case collection")
        suite_case = run_config.suite.case(canonical_case.task_id)
        if (
            canonical_case.case_version != suite_case.case_version
            or canonical_case.digest() != suite_case.canonical_case_digest
            or canonical_case.eval_input().digest() != suite_case.eval_input_digest
            or canonical_case.review_truth.completeness
            is not suite_case.truth_completeness
            or canonical_case.source.suite != run_config.suite.suite_id
        ):
            raise _error("Trial source EvalCase differs from its Suite binding")

        candidate_trial_id = source.trial_id
        if candidate_trial_id is None and source.trial_score is not None:
            candidate_trial_id = source.trial_score.trial_id
        if candidate_trial_id is None and source.submission is not None:
            candidate_trial_id = source.submission.trial_id

        trial_index = source.trial_index
        if trial_index is None and source.trial_score is not None:
            trial_index = source.trial_score.trial_index
        if trial_index is None and candidate_trial_id is not None:
            trial_index = self._find_trial_index(
                run_config,
                canonical_case.task_id,
                candidate_trial_id,
            )
        if trial_index is None:
            raise _error("Trial source cannot be bound without trial_index or trial_id")
        if trial_index > run_config.trial_count:
            raise _error("Trial source trial_index exceeds the Run Trial count")
        expected_trial_id = run_config.trial_id(canonical_case.task_id, trial_index)
        if candidate_trial_id is not None and candidate_trial_id != expected_trial_id:
            raise _error("Trial source ID differs from its immutable Run slot")

        submission = source.submission
        if submission is not None:
            if (
                submission.trial_id != expected_trial_id
                or submission.agent_id != run_config.agent.agent_id
            ):
                raise _error("Submission differs from Run Trial/Agent bindings")

        execution_digest = evaluator_execution.digest()
        if source.review_result is not None:
            if source.review_result.evaluator_execution_digest != execution_digest:
                raise _error("Review result belongs to another evaluator execution")
            for item in (
                *source.review_result.judge_decisions,
                *source.review_result.judge_failures,
                *source.review_result.judge_ungraded,
            ):
                if item.evaluator_execution_digest != execution_digest:
                    raise _error("Review Judge receipt belongs to another execution")
        if source.intent_result is not None:
            for item in (
                *source.intent_result.judge_failures,
                *source.intent_result.judge_ungraded,
            ):
                if item.evaluator_execution_digest != execution_digest:
                    raise _error("Intent Judge receipt belongs to another execution")

        intent_score_result = source.intent_result
        if (
            intent_score_result is not None
            and intent_score_result.status is IntentEvaluationStatus.PENDING_JUDGE
        ):
            intent_score_result = None
        review_score_result = source.review_result
        if (
            review_score_result is not None
            and review_score_result.status is ReviewEvaluationStatus.PENDING_JUDGE
        ):
            review_score_result = None
        replayed_score: Optional[TrialScore] = None
        if submission is not None:
            try:
                replayed_score = TrialScorer.score(
                    scorer,
                    run_config=run_config,
                    evaluator_execution=evaluator_execution,
                    evaluation_revision=evaluation_revision,
                    eval_case=canonical_case,
                    submission=submission,
                    trial_index=trial_index,
                    intent_result=intent_score_result,
                    review_result=review_score_result,
                )
            except (SchemaError, TypeError, ValueError) as exc:
                raise _error(f"Trial score replay failed: {exc}") from exc
        if source.trial_score is not None:
            if replayed_score is None:
                raise _error("persisted Trial score has no terminal evaluator sources")
            if canonical_json_bytes(source.trial_score.to_dict()) != canonical_json_bytes(
                replayed_score.to_dict()
            ):
                raise _error("persisted Trial score differs from source-bound replay")

        score = replayed_score
        if score is not None:
            if score.dimensions != tuple(suite_case.dimensions):
                raise _error("Trial score dimensions differ from Suite dimensions")
            if score.compatibility.metrics_policy != metrics_policy:
                raise _error("Trial score uses another MetricsPolicy")
        return _ResolvedTrial(
            source=source,
            eval_case=canonical_case,
            suite_case=suite_case,
            trial_index=trial_index,
            trial_id=expected_trial_id,
            score=score,
        )

    def _resolve_sources(
        self,
        *,
        run_config: EvalRunConfig,
        evaluator_execution: EvaluatorExecutionConfig,
        evaluation_revision: str,
        cases: Sequence[EvalCase],
        trial_sources: Sequence[TrialEvaluationSource],
        metrics_policy: MetricsPolicy,
        scorer: TrialScorer,
    ) -> Tuple[_ResolvedTrial, ...]:
        values = _sequence(
            trial_sources,
            "TrialEvaluationSource collection",
            MAX_REPORT_TRIALS,
        )
        case_by_task = {item.task_id: item for item in cases}
        resolved = tuple(
            self._resolve_trial(
                run_config=run_config,
                evaluator_execution=evaluator_execution,
                evaluation_revision=evaluation_revision,
                case_by_task=case_by_task,
                source=source,
                metrics_policy=metrics_policy,
                scorer=scorer,
            )
            for source in values
        )
        identities = [(item.eval_case.task_id, item.trial_index) for item in resolved]
        if len(identities) != len(set(identities)):
            raise _error("Trial source collection contains duplicate Run slots")
        trial_ids = [item.trial_id for item in resolved]
        if len(trial_ids) != len(set(trial_ids)):
            raise _error("Trial source collection contains duplicate trial IDs")
        return tuple(
            sorted(
                resolved,
                key=lambda item: (item.eval_case.task_id, item.trial_index),
            )
        )

    @staticmethod
    def _trial_projection(item: _ResolvedTrial) -> Dict[str, Any]:
        source = item.source
        submission = source.submission
        score = item.score
        trace_ref = None
        if submission is not None and submission.trace_ref is not None:
            trace_ref = _safe_trace_ref_projection(submission.trace_ref)
        return {
            "task_id": item.eval_case.task_id,
            "trial_id": item.trial_id,
            "trial_index": item.trial_index,
            "submission_status": (
                None if submission is None else submission.status.value
            ),
            "failure_code": (
                None
                if submission is None or submission.failure is None
                else submission.failure.code.value
            ),
            "submission_digest": (
                None if submission is None else submission.digest()
            ),
            "intent_status": (
                None
                if source.intent_result is None
                else source.intent_result.status.value
            ),
            "intent_result_digest": (
                None
                if source.intent_result is None
                else source.intent_result.digest()
            ),
            "review_status": (
                None
                if source.review_result is None
                else source.review_result.status.value
            ),
            "review_result_digest": (
                None
                if source.review_result is None
                else source.review_result.digest()
            ),
            "score_ref": (
                None
                if score is None
                else _score_ref(score, item.eval_case.task_id, item.trial_id)
            ),
            "score": None if score is None else _safe_score_projection(score),
            "trace_ref": trace_ref,
        }

    def _case_scores(
        self,
        cases: Sequence[EvalCase],
        resolved: Sequence[_ResolvedTrial],
        aggregator: MetricsAggregator,
    ) -> Tuple[Dict[str, CaseScore], Dict[str, Tuple[TrialScore, ...]]]:
        scores_by_task: Dict[str, list[TrialScore]] = defaultdict(list)
        for item in resolved:
            if item.score is not None:
                scores_by_task[item.eval_case.task_id].append(item.score)
        case_scores: Dict[str, CaseScore] = {}
        canonical_trials: Dict[str, Tuple[TrialScore, ...]] = {}
        for case in cases:
            values = tuple(
                sorted(
                    scores_by_task.get(case.task_id, []),
                    key=lambda score: score.trial_index,
                )
            )
            if not values:
                continue
            compatibility = values[0].compatibility
            if any(item.compatibility != compatibility for item in values):
                raise _error("one Case contains incompatible Trial scores")
            try:
                case_score = MetricsAggregator.aggregate_case(
                    aggregator,
                    values,
                    planned_trial_count=values[0].compatibility.trial_count,
                )
            except (TypeError, ValueError) as exc:
                raise _error(f"Case score aggregation failed: {exc}") from exc
            case_scores[case.task_id] = case_score
            canonical_trials[case.task_id] = values
        return case_scores, canonical_trials

    @staticmethod
    def _group_names(
        cases: Sequence[CaseScore],
        requested: Optional[Sequence[str]],
    ) -> Tuple[str, ...]:
        if requested is not None:
            values = _sequence(
                requested,
                "group dimension names",
                MAX_GROUP_DIMENSIONS,
            )
            names = tuple(values)
            if any(type(item) is not str or not item for item in names):
                raise _error("group dimension names contain an invalid value")
            if len(names) != len(set(names)):
                raise _error("group dimension names contain duplicates")
            available = [
                {dimension.name for dimension in case.dimensions}
                for case in cases
            ]
            if any(set(names) - item for item in available):
                raise _error("requested group dimension is missing from a Case")
            return tuple(sorted(names))
        common: Optional[set[str]] = None
        for case in cases:
            names = {dimension.name for dimension in case.dimensions}
            common = names if common is None else common.intersection(names)
        return tuple(sorted(common or set()))

    def _partitions(
        self,
        case_scores: Mapping[str, CaseScore],
        trials_by_task: Mapping[str, Tuple[TrialScore, ...]],
        *,
        group_dimension_names: Optional[Sequence[str]],
        aggregator: MetricsAggregator,
    ) -> Tuple[list[Dict[str, Any]], Dict[str, AggregateScore]]:
        grouped_cases: Dict[str, list[CaseScore]] = defaultdict(list)
        grouped_trials: Dict[str, list[TrialScore]] = defaultdict(list)
        compatibilities: Dict[str, ScoreCompatibilityKey] = {}
        for task_id in sorted(case_scores):
            case_score = case_scores[task_id]
            compatibility_digest = canonical_sha256(
                case_score.compatibility.to_dict()
            )
            existing = compatibilities.get(compatibility_digest)
            if existing is not None and existing != case_score.compatibility:
                raise _error("compatibility digest collision")
            compatibilities[compatibility_digest] = case_score.compatibility
            grouped_cases[compatibility_digest].append(case_score)
            grouped_trials[compatibility_digest].extend(trials_by_task[task_id])

        partitions = []
        aggregates: Dict[str, AggregateScore] = {}
        for compatibility_digest in sorted(grouped_cases):
            cases = tuple(
                sorted(
                    grouped_cases[compatibility_digest],
                    key=lambda item: item.task_id,
                )
            )
            trials = tuple(
                sorted(
                    grouped_trials[compatibility_digest],
                    key=lambda item: (item.task_id, item.trial_index),
                )
            )
            try:
                aggregate = MetricsAggregator.aggregate_cases(
                    aggregator,
                    cases,
                    source_trials=trials,
                )
            except (TypeError, ValueError) as exc:
                raise _error(f"partition aggregation failed: {exc}") from exc
            partition_id = stable_id(
                "report-partition-v1",
                aggregate.compatibility.to_dict(),
            )
            names = self._group_names(cases, group_dimension_names)
            groupings = []
            if names:
                try:
                    grouped_scores = MetricsAggregator.group_case_scores(
                        aggregator,
                        cases,
                        trials,
                        dimension_names=names,
                    )
                except (TypeError, ValueError) as exc:
                    raise _error(f"dimension grouping failed: {exc}") from exc
                groupings.append(
                    {
                        "dimension_names": list(names),
                        "scores": [item.to_dict() for item in grouped_scores],
                    }
                )
            partitions.append(
                {
                    "partition_id": partition_id,
                    "compatibility_digest": compatibility_digest,
                    "compatibility": aggregate.compatibility.to_dict(),
                    "aggregate_score": aggregate.to_dict(),
                    "case_scores": [item.to_dict() for item in cases],
                    "groupings": groupings,
                }
            )
            aggregates[partition_id] = aggregate
        partitions.sort(key=lambda item: item["partition_id"])
        return partitions, aggregates

    @staticmethod
    def _case_records(
        cases: Sequence[EvalCase],
        resolved: Sequence[_ResolvedTrial],
        case_scores: Mapping[str, CaseScore],
        run_config: EvalRunConfig,
    ) -> list[Dict[str, Any]]:
        by_task: Dict[str, list[_ResolvedTrial]] = defaultdict(list)
        for item in resolved:
            by_task[item.eval_case.task_id].append(item)
        records = []
        for case in cases:
            suite_case = run_config.suite.case(case.task_id)
            score = case_scores.get(case.task_id)
            trials = sorted(by_task.get(case.task_id, []), key=lambda item: item.trial_index)
            records.append(
                {
                    "task_id": case.task_id,
                    "case_version": case.case_version,
                    "canonical_case_digest": case.digest(),
                    "eval_input_digest": case.eval_input().digest(),
                    "protocol_id": suite_case.protocol_id,
                    "truth_completeness": suite_case.truth_completeness.value,
                    "dimensions": [item.to_dict() for item in suite_case.dimensions],
                    "planned_trial_count": run_config.trial_count,
                    "terminal_submission_count": sum(
                        item.source.submission is not None for item in trials
                    ),
                    "trial_score_count": sum(item.score is not None for item in trials),
                    "case_score_ref": (
                        None
                        if score is None
                        else _score_ref(score, case.task_id, None)
                    ),
                    "trials": [ReportBuilder._trial_projection(item) for item in trials],
                }
            )
        return records

    @staticmethod
    def _coverage(
        cases: Sequence[EvalCase],
        resolved: Sequence[_ResolvedTrial],
        run_config: EvalRunConfig,
    ) -> Dict[str, Any]:
        expected = {
            run_config.trial_id(case.task_id, index)
            for case in cases
            for index in range(1, run_config.trial_count + 1)
        }
        terminal = [item for item in resolved if item.source.submission is not None]
        terminal_ids = {item.trial_id for item in terminal}
        scored = [item for item in resolved if item.score is not None]
        intent_scored = [
            item
            for item in scored
            if item.score.intent_binding is not None
            and item.score.intent_binding.status is IntentEvaluationStatus.GRADED
        ]
        review_scored = [
            item
            for item in scored
            if item.score.review_binding is not None
            and item.score.review_binding.status is ReviewEvaluationStatus.GRADED
        ]
        fully_scored = [
            item
            for item in scored
            if item.score.intent_binding is not None
            and item.score.intent_binding.status is IntentEvaluationStatus.GRADED
            and item.score.review_binding is not None
            and item.score.review_binding.status is ReviewEvaluationStatus.GRADED
        ]
        unevaluated = sorted(
            item.trial_id
            for item in terminal
            if item.score is None
        )
        return {
            "planned_case_count": len(cases),
            "planned_trial_count": len(expected),
            "represented_trial_count": len(resolved),
            "terminal_submission_count": len(terminal),
            "trial_score_count": len(scored),
            "intent_scored_trial_count": len(intent_scored),
            "review_scored_trial_count": len(review_scored),
            "fully_scored_trial_count": len(fully_scored),
            "submission_status_breakdown": _breakdown(
                item.source.submission.status.value for item in terminal
            ),
            "intent_status_breakdown": _breakdown(
                (
                    "missing"
                    if item.source.intent_result is None
                    else item.source.intent_result.status.value
                )
                for item in terminal
            ),
            "review_status_breakdown": _breakdown(
                (
                    "missing"
                    if item.source.review_result is None
                    else item.source.review_result.status.value
                )
                for item in terminal
            ),
            "failure_code_breakdown": _breakdown(
                item.source.submission.failure.code.value
                for item in terminal
                if item.source.submission.failure is not None
            ),
            "nonterminal_trial_ids": sorted(expected - terminal_ids),
            "unevaluated_terminal_trial_ids": unevaluated,
        }

    @staticmethod
    def _agent_failure(item: _ResolvedTrial) -> Optional[Dict[str, Any]]:
        submission = item.source.submission
        if submission is None or submission.status is SubmissionStatus.COMPLETED:
            return None
        failure = submission.failure
        return {
            "task_id": item.eval_case.task_id,
            "trial_id": item.trial_id,
            "status": submission.status.value,
            "failure_code": None if failure is None else failure.code.value,
            "retryable": None if failure is None else failure.retryable,
            "message_digest": (
                None
                if failure is None
                else canonical_sha256({"message": failure.message})
            ),
        }

    @staticmethod
    def _truth_miss_records(item: _ResolvedTrial) -> list[Dict[str, Any]]:
        severe = {
            FindingSeverity.HIGH,
            FindingSeverity.CRITICAL,
        }
        truths = [
            truth
            for truth in item.eval_case.review_truth.expected_findings
            if truth.required and truth.severity in severe
        ]
        if not truths:
            return []
        if item.source.submission is None:
            return []
        result = item.source.review_result
        source_status = "graded"
        reason = "required_truth_unmatched"
        matched_ids: set[str] = set()
        if result is not None and result.status is ReviewEvaluationStatus.GRADED:
            matched_ids = {
                outcome.matched_expected_truth_id
                for outcome in result.finding_outcomes
                if outcome.matched_expected_truth_id is not None
                and outcome.issue_judgement is IssueJudgement.CONFIRMED
                and outcome.disposition is FindingDisposition.MATCHED
            }
        elif result is not None:
            # A pending/ungraded Judge is not silently turned into a factual
            # miss; it has its own diagnostic bucket below.
            return []
        else:
            score = item.score
            contribution = None
            if score is not None:
                contribution = score.contribution(CoreMetric.ISSUE_RECALL)
            if (
                contribution is not None
                and contribution.source_status
                is MetricSourceStatus.FAILURE_AS_MISS
            ):
                source_status = "failure_as_miss"
                reason = "agent_failure"
            else:
                return []
        records = []
        for truth in truths:
            if truth.truth_id in matched_ids:
                continue
            records.append(
                {
                    "task_id": item.eval_case.task_id,
                    "trial_id": item.trial_id,
                    "truth_id": truth.truth_id,
                    "severity": truth.severity.value,
                    "category": truth.category,
                    "claim": truth.claim,
                    "required": truth.required,
                    "required_context_level": truth.required_context_level.value,
                    "source_status": source_status,
                    "reason_codes": [reason],
                }
            )
        return records

    @staticmethod
    def _fabricated_records(item: _ResolvedTrial) -> list[Dict[str, Any]]:
        result = item.source.review_result
        submission = item.source.submission
        if (
            result is None
            or result.status is not ReviewEvaluationStatus.GRADED
            or submission is None
            or submission.review is None
        ):
            return []
        findings = {finding.finding_id: finding for finding in submission.review.findings}
        records = []
        for outcome in result.finding_outcomes:
            if outcome.issue_judgement is not IssueJudgement.FABRICATED:
                continue
            finding = findings.get(outcome.finding_id)
            records.append(
                {
                    "task_id": item.eval_case.task_id,
                    "trial_id": item.trial_id,
                    "finding_id": outcome.finding_id,
                    "finding": (
                        None
                        if finding is None
                        else _redact_sensitive_structure(finding.to_dict())
                    ),
                    "issue_judgement": outcome.issue_judgement.value,
                    "disposition": outcome.disposition.value,
                    "known_invalid_truth_id": outcome.matched_known_invalid_truth_id,
                    "evidence_integrity": outcome.evidence_integrity.value,
                    "evidence_support": outcome.evidence_support.value,
                    "strict_publishable": outcome.strict_publishable,
                    "reason_codes": [item.value for item in outcome.reason_codes],
                }
            )
        return sorted(records, key=lambda item: (item["task_id"], item["trial_id"], item["finding_id"]))

    @staticmethod
    def _novel_disallowed_records(item: _ResolvedTrial) -> list[Dict[str, Any]]:
        result = item.source.review_result
        submission = item.source.submission
        if result is None or submission is None or submission.review is None:
            return []
        findings = {finding.finding_id: finding for finding in submission.review.findings}
        records = []
        for outcome in result.finding_outcomes:
            if outcome.disposition is not FindingDisposition.NOVEL_DISALLOWED:
                continue
            finding = findings.get(outcome.finding_id)
            records.append(
                {
                    "task_id": item.eval_case.task_id,
                    "trial_id": item.trial_id,
                    "finding_id": outcome.finding_id,
                    "finding": (
                        None
                        if finding is None
                        else _redact_sensitive_structure(finding.to_dict())
                    ),
                    "disposition": outcome.disposition.value,
                    "issue_judgement": outcome.issue_judgement.value,
                    "reason_codes": [code.value for code in outcome.reason_codes],
                }
            )
        return sorted(
            records,
            key=lambda item: (item["task_id"], item["trial_id"], item["finding_id"]),
        )

    @staticmethod
    def _ungraded_records(item: _ResolvedTrial) -> list[Dict[str, Any]]:
        records = []
        if item.source.intent_result is not None:
            result = item.source.intent_result
            if result.status is IntentEvaluationStatus.UNGRADED:
                records.append(
                    {
                        "task_id": item.eval_case.task_id,
                        "trial_id": item.trial_id,
                        "phase": "intent",
                        "status": result.status.value,
                        "result_digest": result.digest(),
                    }
                )
        if item.source.review_result is not None:
            result = item.source.review_result
            if result.status is ReviewEvaluationStatus.UNGRADED:
                records.append(
                    {
                        "task_id": item.eval_case.task_id,
                        "trial_id": item.trial_id,
                        "phase": "review",
                        "status": result.status.value,
                        "result_digest": result.digest(),
                    }
                )
        return records

    @staticmethod
    def _pending_records(item: _ResolvedTrial) -> list[Dict[str, Any]]:
        records = []
        for phase, result, status in (
            (
                "intent",
                item.source.intent_result,
                IntentEvaluationStatus.PENDING_JUDGE,
            ),
            (
                "review",
                item.source.review_result,
                ReviewEvaluationStatus.PENDING_JUDGE,
            ),
        ):
            if result is not None and result.status is status:
                records.append(
                    {
                        "task_id": item.eval_case.task_id,
                        "trial_id": item.trial_id,
                        "phase": phase,
                        "status": result.status.value,
                        "result_digest": result.digest(),
                    }
                )
        return records

    @staticmethod
    def _missing_evaluation_records(item: _ResolvedTrial) -> list[Dict[str, Any]]:
        submission = item.source.submission
        if submission is None:
            return []
        records = []
        if (
            submission.intent is not None
            and item.source.intent_result is None
            and item.eval_case.intent_truth.scorable
        ):
            records.append(
                {
                    "task_id": item.eval_case.task_id,
                    "trial_id": item.trial_id,
                    "phase": "intent",
                    "status": "missing_evaluation",
                }
            )
        if submission.review is not None and item.source.review_result is None:
            records.append(
                {
                    "task_id": item.eval_case.task_id,
                    "trial_id": item.trial_id,
                    "phase": "review",
                    "status": "missing_evaluation",
                }
            )
        return records

    @staticmethod
    def _nonterminal_records(item: _ResolvedTrial) -> list[Dict[str, Any]]:
        if item.source.submission is not None:
            return []
        receipts = [_object(value, "Trial timeline receipt") for value in item.source.timeline]
        return [
            {
                "task_id": item.eval_case.task_id,
                "trial_id": item.trial_id,
                "status": "nonterminal",
                "receipt_ids": sorted(
                    value.get("receipt_id")
                    for value in receipts
                    if value.get("receipt_id") is not None
                ),
                "stages": sorted(
                    value.get("stage")
                    for value in receipts
                    if value.get("stage") is not None
                ),
                "reason": "terminal_submission_missing",
            }
        ]

    @staticmethod
    def _missing_planned_records(
        cases: Sequence[EvalCase],
        resolved: Sequence[_ResolvedTrial],
        run_config: EvalRunConfig,
    ) -> list[Dict[str, Any]]:
        represented = {item.trial_id for item in resolved}
        records = []
        for case in cases:
            for trial_index in range(1, run_config.trial_count + 1):
                trial_id = run_config.trial_id(case.task_id, trial_index)
                if trial_id in represented:
                    continue
                records.append(
                    {
                        "task_id": case.task_id,
                        "trial_id": trial_id,
                        "status": "unrepresented",
                        "receipt_ids": [],
                        "stages": [],
                        "reason": "trial_source_missing",
                    }
                )
        return records

    @staticmethod
    def _judge_diagnostics(
        item: _ResolvedTrial,
    ) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
        failures: list[Dict[str, Any]] = []
        ungraded: list[Dict[str, Any]] = []
        unknowns: list[Dict[str, Any]] = []
        for phase, evaluation in (
            ("intent", item.source.intent_result),
            ("review", item.source.review_result),
        ):
            if evaluation is None:
                continue
            result_digest = evaluation.digest()
            for receipt in evaluation.judge_failures:
                record = _judge_reference(
                    receipt,
                    phase=phase,
                    kind="failure",
                    parent_result_digest=result_digest,
                )
                record.update({"task_id": item.eval_case.task_id, "trial_id": item.trial_id})
                failures.append(record)
            for receipt in evaluation.judge_ungraded:
                record = _judge_reference(
                    receipt,
                    phase=phase,
                    kind="ungraded",
                    parent_result_digest=result_digest,
                )
                record.update({"task_id": item.eval_case.task_id, "trial_id": item.trial_id})
                ungraded.append(record)
            for decision in evaluation.judge_decisions:
                unknown = _semantic_unknown(decision)
                if unknown is not None:
                    unknown.update(
                        {
                            "phase": phase,
                            "task_id": item.eval_case.task_id,
                            "trial_id": item.trial_id,
                            "parent_result_digest": result_digest,
                        }
                    )
                    unknowns.append(unknown)
        key = lambda record: (
            record.get("task_id", ""),
            record.get("trial_id", ""),
            record.get("phase", ""),
            record.get("request_id", ""),
            record.get("kind", ""),
        )
        return sorted(failures, key=key), sorted(ungraded, key=key), sorted(unknowns, key=key)

    @staticmethod
    def _usage_missing(
        scored: Sequence[_ResolvedTrial],
    ) -> Dict[str, list[str]]:
        fields = (
            "elapsed_seconds",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "tool_calls",
            "cost_amount",
        )
        result = {field: [] for field in fields}
        for item in scored:
            if item.score is None:
                continue
            usage = item.score.usage
            for field in fields:
                if getattr(usage, field) is None:
                    result[field].append(item.trial_id)
        return {field: sorted(values) for field, values in result.items()}

    @staticmethod
    def _trace_refs(resolved: Sequence[_ResolvedTrial]) -> list[Dict[str, Any]]:
        records = []
        for item in resolved:
            submission = item.source.submission
            if submission is None or submission.trace_ref is None:
                continue
            records.append(
                {
                    "task_id": item.eval_case.task_id,
                    "trial_id": item.trial_id,
                    "trace_ref": _safe_trace_ref_projection(submission.trace_ref),
                }
            )
        return sorted(records, key=lambda item: (item["task_id"], item["trial_id"]))

    @staticmethod
    def _coerce_run_manifest(value: Any) -> RunManifest:
        if type(value) is RunManifest:
            return value
        try:
            return RunManifest.from_dict(_object(value, "RunManifest"))
        except (SchemaError, TypeError, ValueError) as exc:
            raise _error(f"RunManifest is invalid: {exc}") from exc

    @staticmethod
    def _validate_trial_manifest_binding(
        run_config: EvalRunConfig,
        run_manifest: RunManifest,
        item: _ResolvedTrial,
    ) -> None:
        trial_manifest = item.source.trial_manifest
        if type(trial_manifest) is not TrialManifest:
            raise _error("RunManifest-backed report requires every TrialManifest")
        if (
            trial_manifest.run_id != run_config.run_id
            or trial_manifest.task_id != item.eval_case.task_id
            or trial_manifest.trial_id != item.trial_id
            or trial_manifest.trial_index != item.trial_index
            or trial_manifest.canonical_case_digest != item.eval_case.digest()
            or trial_manifest.eval_input_digest != item.eval_case.eval_input().digest()
            or trial_manifest.agent_config_digest != run_config.agent_config_digest
        ):
            raise _error("TrialManifest does not bind the supplied Trial sources")
        if (
            trial_manifest.initial_evaluator_execution_digest
            != run_manifest.initial_evaluator_execution_digest
        ):
            raise _error(
                "TrialManifest initial evaluator binding differs from RunManifest"
            )
        plans = tuple(
            plan for plan in run_manifest.trials if plan.trial_id == item.trial_id
        )
        if len(plans) != 1:
            raise _error("RunManifest has no unique plan for TrialManifest")
        plan = plans[0]
        trial_manifest_bytes = canonical_json_bytes(trial_manifest.to_dict())
        if (
            plan.task_id != item.eval_case.task_id
            or plan.trial_index != item.trial_index
            or plan.canonical_case_digest != item.eval_case.digest()
            or plan.eval_input_digest != item.eval_case.eval_input().digest()
            or plan.manifest.sha256 != canonical_sha256(trial_manifest.to_dict())
            or plan.manifest.size_bytes != len(trial_manifest_bytes)
        ):
            raise _error("RunManifest Trial plan differs from TrialManifest")

    def _validate_summary_manifests(
        self,
        run_config: EvalRunConfig,
        run_manifest: RunManifest,
        resolved: Sequence[_ResolvedTrial],
    ) -> None:
        if {item.trial_id for item in resolved} != {
            plan.trial_id for plan in run_manifest.trials
        }:
            raise _error(
                "RunManifest-backed summary requires all planned Trial sources"
            )
        for item in resolved:
            self._validate_trial_manifest_binding(
                run_config,
                run_manifest,
                item,
            )

    def _diagnostics(
        self,
        cases: Sequence[EvalCase],
        resolved: Sequence[_ResolvedTrial],
        aggregates: Mapping[str, AggregateScore],
        run_config: EvalRunConfig,
    ) -> Dict[str, Any]:
        agent_failures = []
        severe_misses = []
        fabricated = []
        novel_disallowed = []
        ungraded = []
        pending = []
        missing_evaluations = []
        nonterminal = []
        judge_failures = []
        judge_ungraded = []
        semantic_unknowns = []
        clarification_receipts = []
        evidence_diagnostics = []
        for item in resolved:
            failure = self._agent_failure(item)
            if failure is not None:
                agent_failures.append(failure)
            severe_misses.extend(self._truth_miss_records(item))
            fabricated.extend(self._fabricated_records(item))
            novel_disallowed.extend(self._novel_disallowed_records(item))
            ungraded.extend(self._ungraded_records(item))
            pending.extend(self._pending_records(item))
            missing_evaluations.extend(self._missing_evaluation_records(item))
            nonterminal.extend(self._nonterminal_records(item))
            failures, judge_ungraded_items, unknowns = self._judge_diagnostics(item)
            judge_failures.extend(failures)
            judge_ungraded.extend(judge_ungraded_items)
            semantic_unknowns.extend(unknowns)
            if item.source.intent_result is not None:
                clarification = _clarification_projection(item.source.intent_result)
                if clarification is not None:
                    clarification_receipts.append(
                        {
                            "task_id": item.eval_case.task_id,
                            "trial_id": item.trial_id,
                            **clarification,
                        }
                    )
            for evidence in _evidence_projection(item.source.review_result):
                evidence_diagnostics.append(
                    {
                        "task_id": item.eval_case.task_id,
                        "trial_id": item.trial_id,
                        **evidence,
                    }
                )
        nonterminal.extend(
            self._missing_planned_records(cases, resolved, run_config)
        )
        usage_missing_trial_ids = self._usage_missing(resolved)
        usage_missing_coverage = []
        for partition_id, aggregate in sorted(aggregates.items()):
            usage_missing_coverage.append(
                {
                    "partition_id": partition_id,
                    "usage": aggregate.usage.to_dict(),
                }
            )
        sort_key = lambda item: canonical_json(item)
        for values in (
            agent_failures,
            severe_misses,
            fabricated,
            novel_disallowed,
            ungraded,
            pending,
            missing_evaluations,
            nonterminal,
            judge_failures,
            judge_ungraded,
            semantic_unknowns,
            clarification_receipts,
            evidence_diagnostics,
        ):
            values.sort(key=sort_key)
        return {
            "agent_failures": agent_failures,
            "ungraded_trials": ungraded,
            "pending_evaluations": pending,
            "missing_evaluations": missing_evaluations,
            "nonterminal_trials": nonterminal,
            "critical_high_misses": severe_misses,
            "fabricated_findings": fabricated,
            "novel_disallowed_findings": novel_disallowed,
            "judge_failures": judge_failures,
            "judge_ungraded": judge_ungraded,
            "semantic_unknowns": semantic_unknowns,
            "evidence_diagnostics": evidence_diagnostics,
            "clarification_receipts": clarification_receipts,
            "trace_refs": self._trace_refs(resolved),
            "usage_missing_trial_ids": usage_missing_trial_ids,
            "usage_missing_coverage": usage_missing_coverage,
            "metric_diagnostics": [
                {
                    "partition_id": partition_id,
                    "metrics": [item.to_dict() for item in aggregate.metrics],
                }
                for partition_id, aggregate in sorted(aggregates.items())
            ],
        }

    def _source_bindings(
        self,
        run_config: EvalRunConfig,
        evaluator_execution: EvaluatorExecutionConfig,
        evaluation_revision: str,
        evaluation_id: str,
        *,
        run_manifest: Optional[Any],
        metrics_policy: MetricsPolicy,
    ) -> Dict[str, Any]:
        run_manifest_digest = None
        if run_manifest is not None:
            if type(run_manifest) is not RunManifest:
                try:
                    run_manifest = RunManifest.from_dict(
                        _object(run_manifest, "RunManifest")
                    )
                except (SchemaError, TypeError, ValueError) as exc:
                    raise _error(f"RunManifest is invalid: {exc}") from exc
            manifest_payload = run_manifest.to_dict()
            if manifest_payload.get("run_id") != run_config.run_id:
                raise _error("RunManifest belongs to another Run")
            if (
                run_manifest.run_config.sha256 != run_config.digest()
                or run_manifest.case_snapshot.sha256
                != run_config.suite.case_snapshot_digest
                or run_manifest.agent_config_digest
                != run_config.agent_config_digest
                or run_manifest.initial_evaluator_execution_digest
                != EvaluatorExecutionConfig.from_resource_budgets(
                    run_config.evaluator,
                    run_config.resource_budgets,
                ).digest()
            ):
                raise _error("RunManifest does not bind the supplied Run sources")
            expected_plans = {
                run_config.trial_id(case.task_id, trial_index): (
                    case.task_id,
                    trial_index,
                    case.canonical_case_digest,
                    case.eval_input_digest,
                )
                for case in run_config.suite.cases
                for trial_index in range(1, run_config.trial_count + 1)
            }
            actual_plan_ids = {plan.trial_id for plan in run_manifest.trials}
            if actual_plan_ids != set(expected_plans) or len(run_manifest.trials) != len(
                expected_plans
            ):
                raise _error("RunManifest Trial plan coverage differs from RunConfig")
            for plan in run_manifest.trials:
                task_id, trial_index, case_digest, input_digest = expected_plans[
                    plan.trial_id
                ]
                case_path_id = derive_case_path_id(task_id)
                expected_trial_manifest = TrialManifest(
                    schema_version=TrialManifest.SCHEMA_VERSION,
                    run_id=run_config.run_id,
                    task_id=task_id,
                    case_path_id=case_path_id,
                    canonical_case_digest=case_digest,
                    eval_input_digest=input_digest,
                    trial_id=plan.trial_id,
                    trial_index=trial_index,
                    seed=derive_trial_seed(
                        run_config.run_id,
                        task_id,
                        trial_index,
                    ),
                    agent_config_digest=run_config.agent_config_digest,
                    initial_evaluator_execution_digest=(
                        run_manifest.initial_evaluator_execution_digest
                    ),
                )
                expected_manifest_bytes = canonical_json_bytes(
                    expected_trial_manifest.to_dict()
                )
                if (
                    plan.task_id != task_id
                    or plan.case_path_id != case_path_id
                    or plan.trial_index != trial_index
                    or plan.canonical_case_digest != case_digest
                    or plan.eval_input_digest != input_digest
                    or plan.manifest.sha256
                    != canonical_sha256(expected_trial_manifest.to_dict())
                    or plan.manifest.size_bytes != len(expected_manifest_bytes)
                ):
                    raise _error("RunManifest Trial plan differs from RunConfig")
            run_manifest_digest = _object_digest(run_manifest, "RunManifest")
        return {
            "run_id": run_config.run_id,
            "run_config_digest": run_config.digest(),
            "run_manifest_digest": run_manifest_digest,
            "case_snapshot_id": run_config.suite.case_snapshot_id,
            "case_snapshot_digest": run_config.suite.case_snapshot_digest,
            "evaluation_id": evaluation_id,
            "evaluation_revision": evaluation_revision,
            "evaluator_execution_digest": evaluator_execution.digest(),
            "metrics_policy": metrics_policy.to_dict(),
        }

    def build_summary(
        self,
        run_config: EvalRunConfig,
        evaluator_execution: EvaluatorExecutionConfig,
        evaluation_revision: str,
        eval_cases: Optional[Sequence[EvalCase]] = None,
        trial_sources: Optional[Sequence[TrialEvaluationSource]] = None,
        *,
        cases: Optional[Sequence[EvalCase]] = None,
        sources: Optional[Sequence[TrialEvaluationSource]] = None,
        group_dimension_names: Optional[Sequence[str]] = None,
        run_manifest: Optional[Any] = None,
    ) -> RunReportSummary:
        """Build the authoritative, compatibility-partitioned Run summary."""

        metrics_policy, scorer, aggregator = self._trusted_components()
        revision, evaluation_id = self._validate_run_sources(
            run_config,
            evaluator_execution,
            evaluation_revision,
        )
        if eval_cases is not None and cases is not None:
            raise _error("provide either eval_cases or cases, not both")
        if trial_sources is not None and sources is not None:
            raise _error("provide either trial_sources or sources, not both")
        actual_cases = eval_cases if eval_cases is not None else cases
        actual_sources = trial_sources if trial_sources is not None else sources
        if actual_cases is None:
            raise _error("build_summary requires the real EvalCase collection")
        if actual_sources is None:
            actual_sources = ()

        canonical_cases = self._normalize_cases(run_config, actual_cases)
        resolved = self._resolve_sources(
            run_config=run_config,
            evaluator_execution=evaluator_execution,
            evaluation_revision=revision,
            cases=canonical_cases,
            trial_sources=actual_sources,
            metrics_policy=metrics_policy,
            scorer=scorer,
        )
        case_scores, trials_by_task = self._case_scores(
            canonical_cases,
            resolved,
            aggregator,
        )
        partitions, aggregates = self._partitions(
            case_scores,
            trials_by_task,
            group_dimension_names=group_dimension_names,
            aggregator=aggregator,
        )
        coverage = self._coverage(canonical_cases, resolved, run_config)
        case_records = self._case_records(
            canonical_cases,
            resolved,
            case_scores,
            run_config,
        )
        canonical_run_manifest = None
        if run_manifest is not None:
            canonical_run_manifest = self._coerce_run_manifest(run_manifest)
            self._validate_summary_manifests(
                run_config,
                canonical_run_manifest,
                resolved,
            )
        source_bindings = self._source_bindings(
            run_config,
            evaluator_execution,
            revision,
            evaluation_id,
            run_manifest=canonical_run_manifest,
            metrics_policy=metrics_policy,
        )
        identity = {
            "suite": {
                "suite_id": run_config.suite.suite_id,
                "suite_version": run_config.suite.suite_version,
                "manifest_digest": run_config.suite.manifest_digest,
                "case_snapshot_id": run_config.suite.case_snapshot_id,
                "case_snapshot_digest": run_config.suite.case_snapshot_digest,
            },
            "agent": run_config.agent.to_dict(),
            "evaluator": {
                "configuration": evaluator_execution.evaluator.to_dict(),
                "evaluator_config_digest": evaluator_execution.evaluator_config_digest,
                "execution_config_digest": evaluator_execution.digest(),
                "evaluation_id": evaluation_id,
                "evaluation_revision": revision,
            },
        }
        body: Dict[str, Any] = {
            "schema_version": RUN_REPORT_SUMMARY_SCHEMA_VERSION,
            "report_revision": REPORT_REVISION,
            "source_bindings": source_bindings,
            "identity": identity,
            "coverage": coverage,
            "partitions": partitions,
            "cases": case_records,
            "diagnostics": self._diagnostics(
                canonical_cases,
                resolved,
                aggregates,
                run_config,
            ),
        }
        payload = {
            **body,
            "summary_id": stable_id("run-report-summary-v1", body),
        }
        return RunReportSummary._seal(payload, _token=_SUMMARY_SEAL_TOKEN)

    def build_inspection(
        self,
        run_config: EvalRunConfig,
        evaluator_execution: EvaluatorExecutionConfig,
        evaluation_revision: str,
        trial_source: Optional[TrialEvaluationSource] = None,
        *,
        source: Optional[TrialEvaluationSource] = None,
        run_manifest: Optional[Any] = None,
    ) -> TrialInspection:
        """Build one audit projection without embedding trace content or secrets."""

        metrics_policy, scorer, _aggregator = self._trusted_components()
        revision, evaluation_id = self._validate_run_sources(
            run_config,
            evaluator_execution,
            evaluation_revision,
        )
        if trial_source is not None and source is not None:
            raise _error("provide either trial_source or source, not both")
        actual_source = trial_source if trial_source is not None else source
        if type(actual_source) is not TrialEvaluationSource:
            raise _error("build_inspection requires TrialEvaluationSource")
        resolved = self._resolve_trial(
            run_config=run_config,
            evaluator_execution=evaluator_execution,
            evaluation_revision=revision,
            case_by_task={actual_source.eval_case.task_id: actual_source.eval_case},
            source=actual_source,
            metrics_policy=metrics_policy,
            scorer=scorer,
        )
        canonical_run_manifest = None
        if run_manifest is not None:
            canonical_run_manifest = self._coerce_run_manifest(run_manifest)
            self._validate_trial_manifest_binding(
                run_config,
                canonical_run_manifest,
                resolved,
            )
        elif actual_source.trial_manifest is not None:
            raise _error(
                "TrialManifest inspection requires its immutable RunManifest"
            )
        source_bindings = {
            **self._source_bindings(
                run_config,
                evaluator_execution,
                revision,
                evaluation_id,
                run_manifest=canonical_run_manifest,
                metrics_policy=metrics_policy,
            ),
            "task_id": resolved.eval_case.task_id,
            "trial_id": resolved.trial_id,
            "trial_index": resolved.trial_index,
            "canonical_case_digest": resolved.eval_case.digest(),
            "eval_input_digest": resolved.eval_case.eval_input().digest(),
            "submission_digest": (
                None
                if actual_source.submission is None
                else actual_source.submission.digest()
            ),
            "intent_result_digest": (
                None
                if actual_source.intent_result is None
                else actual_source.intent_result.digest()
            ),
            "review_result_digest": (
                None
                if actual_source.review_result is None
                else actual_source.review_result.digest()
            ),
            "trial_score_digest": (
                None if resolved.score is None else resolved.score.digest()
            ),
        }
        timeline = _timeline_projection(actual_source.timeline)
        for receipt in timeline:
            if receipt.get("run_id") not in (None, run_config.run_id):
                raise _error("timeline receipt belongs to another Run")
            if receipt.get("task_id") not in (None, resolved.eval_case.task_id):
                raise _error("timeline receipt belongs to another Case")
            if receipt.get("trial_id") not in (None, resolved.trial_id):
                raise _error("timeline receipt belongs to another Trial")
        trial_manifest = None
        if actual_source.trial_manifest is not None:
            if type(actual_source.trial_manifest) is not TrialManifest:
                raise _error("Trial source TrialManifest was not canonicalized")
            trial_manifest = actual_source.trial_manifest.to_dict()

        trace_ref = None
        if (
            actual_source.submission is not None
            and actual_source.submission.trace_ref is not None
        ):
            trace_ref = _safe_trace_ref_projection(actual_source.submission.trace_ref)
        body: Dict[str, Any] = {
            "schema_version": TRIAL_INSPECTION_SCHEMA_VERSION,
            "report_revision": REPORT_REVISION,
            "source_bindings": source_bindings,
            "trial_manifest": trial_manifest,
            "timeline": timeline,
            "input": _safe_eval_input_projection(resolved.eval_case.eval_input()),
            "submission": (
                None
                if actual_source.submission is None
                else _safe_submission_projection(actual_source.submission)
            ),
            "score": (
                None
                if resolved.score is None
                else _safe_score_projection(resolved.score)
            ),
            "intent_evaluation": (
                None
                if actual_source.intent_result is None
                else _safe_intent_evaluation_projection(actual_source.intent_result)
            ),
            "review_evaluation": (
                None
                if actual_source.review_result is None
                else _safe_review_evaluation_projection(actual_source.review_result)
            ),
            "judge_artifact_refs": _judge_artifact_refs(
                actual_source.intent_result,
                actual_source.review_result,
            ),
            "clarification_match_receipts": _clarification_projection(
                actual_source.intent_result
            ),
            "evidence_diagnostics": _evidence_projection(
                actual_source.review_result
            ),
            "trace": {
                "trace_ref": trace_ref,
                "capture": _trace_capture_projection(
                    actual_source.trace_capture,
                    maximum_total_bytes=run_config.resource_budgets.max_trace_bytes,
                    maximum_file_bytes=run_config.resource_budgets.max_trace_bytes,
                ),
            },
        }
        payload = {
            **body,
            "inspection_id": stable_id("trial-inspection-v1", body),
        }
        return TrialInspection._seal(payload, _token=_INSPECTION_SEAL_TOKEN)

    def hydrate_summary(self, value: Any, **sources: Any) -> RunReportSummary:
        return RunReportSummary.from_dict(value, builder=self, **sources)

    def hydrate_inspection(self, value: Any, **sources: Any) -> TrialInspection:
        return TrialInspection.from_dict(value, builder=self, **sources)


_INTENT_REPORT_METRICS = {
    "intent_claim_precision",
    "intent_claim_recall",
    "intent_partially_supported_rate",
    "intent_unsupported_rate",
    "intent_contradicted_rate",
    "intent_unknown_rate",
    "clarification_accuracy",
    "intent_case_pass_rate",
}
_REVIEW_REPORT_METRICS = {
    "issue_precision",
    "issue_recall",
    "issue_f1",
    "severity_weighted_recall",
    "critical_high_miss_count",
    "fabricated_findings_per_pr",
    "fabricated_rate",
    "plausible_rate",
    "review_unknown_rate",
    "line_precision",
    "line_recall",
    "evidence_validity",
    "evidence_support_rate",
    "publishable_finding_precision",
}
_RELIABILITY_REPORT_METRICS = {
    "agent_failure_rate",
    "judge_failure_rate",
    "judge_ungraded_rate",
    "judge_semantic_unknown_rate",
}


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value).replace("|", "\\|").replace("\r", "").replace("\n", " ")
    return text if text else ""


def _coverage_text(coverage: Any) -> str:
    if coverage is None:
        return "null (derived metric; see source coverages)"
    return canonical_json(coverage)


def _metric_value_text(metric: Mapping[str, Any]) -> str:
    value = metric.get("value_ppm")
    if value is None:
        return "null"
    return f"{value} ppm"


def _render_metric_table(
    lines: list[str],
    metrics: Sequence[Mapping[str, Any]],
    *,
    title: str,
) -> None:
    lines.extend(
        [
            f"### {title}",
            "",
            "| Metric | Kind | Numerator | Denominator | Value | Null reason | Coverage |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for metric in sorted(metrics, key=lambda item: str(item.get("metric", ""))):
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(metric.get("metric")),
                    _markdown_cell(metric.get("kind")),
                    _markdown_cell(metric.get("numerator")),
                    _markdown_cell(metric.get("denominator")),
                    _markdown_cell(_metric_value_text(metric)),
                    _markdown_cell(metric.get("null_reason")),
                    _markdown_cell(_coverage_text(metric.get("coverage"))),
                )
            )
            + " |"
        )
        derived = metric.get("derived_coverages") or []
        if derived:
            for component in derived:
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            _markdown_cell(
                                f"{metric.get('metric')} derived coverage[{component.get('metric')}]"
                            ),
                            "derived",
                            "-",
                            "-",
                            "-",
                            "-",
                            _markdown_cell(_coverage_text(component.get("coverage"))),
                        )
                    )
                    + " |"
                )
    lines.append("")


def _render_key_value_list(lines: list[str], values: Mapping[str, Any]) -> None:
    for key in sorted(values):
        value = values[key]
        if isinstance(value, (dict, list)):
            rendered = canonical_json(value)
        else:
            rendered = _markdown_cell(value)
        lines.append(f"- `{key}`: {rendered}")


def _render_diagnostic_table(
    lines: list[str],
    title: str,
    records: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    lines.extend([f"### {title}", ""])
    if not records:
        lines.extend(["None.", ""])
        return
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for record in sorted(records, key=canonical_json):
        lines.append(
            "| "
            + " | ".join(_markdown_cell(record.get(column)) for column in columns)
            + " |"
        )
    lines.append("")


def render_run_markdown(summary: RunReportSummary) -> str:
    """Render a deterministic Markdown view of an already-built summary.

    This function intentionally accepts no source objects.  It only reads the
    canonical summary projection; all numerators, denominators, null reasons,
    and coverage values come verbatim from ``summary.json``.
    """

    if type(summary) is not RunReportSummary:
        raise _error("render_run_markdown requires RunReportSummary")
    payload = summary.to_dict()
    lines = [
        "# Code Review Evaluation Report",
        "",
        f"- Summary ID: `{payload['summary_id']}`",
        f"- Report revision: `{payload['report_revision']}`",
        "",
        "## Run coverage",
        "",
    ]
    _render_key_value_list(lines, payload["coverage"])
    lines.extend(["", "## Provenance", ""])
    identity = payload["identity"]
    suite = identity.get("suite", {})
    evaluator = identity.get("evaluator", {})
    lines.append(f"- Suite: `{suite.get('suite_id')}` / `{suite.get('suite_version')}`")
    lines.append(f"- Agent: `{identity.get('agent', {}).get('agent_id')}`")
    lines.append(f"- Evaluation: `{evaluator.get('evaluation_id')}`")
    lines.append(
        "- Evaluator execution digest: `"
        + str(payload["source_bindings"].get("evaluator_execution_digest"))
        + "`"
    )
    lines.append("")

    partitions = payload["partitions"]
    if not partitions:
        lines.extend(
            [
                "## Metrics partitions",
                "",
                "No terminal Trial has a score yet; quality metrics are unavailable.",
                "",
            ]
        )
    else:
        lines.extend(["## Metrics partitions", ""])
        for index, partition in enumerate(partitions, start=1):
            compatibility = partition.get("compatibility", {})
            lines.extend(
                [
                    f"### Partition {index}: `{partition.get('partition_id')}`",
                    "",
                    f"- Truth completeness: `{compatibility.get('truth_completeness')}`",
                    f"- Protocol: `{compatibility.get('protocol_id')}`",
                    f"- Novel Finding policy: `{compatibility.get('novel_finding_policy')}`",
                    "",
                ]
            )
            metric_values = partition.get("aggregate_score", {}).get("metrics", [])
            _render_metric_table(
                lines,
                [item for item in metric_values if item.get("metric") in _INTENT_REPORT_METRICS],
                title="Intent",
            )
            _render_metric_table(
                lines,
                [item for item in metric_values if item.get("metric") in _REVIEW_REPORT_METRICS],
                title="Review and Evidence",
            )
            _render_metric_table(
                lines,
                [item for item in metric_values if item.get("metric") in _RELIABILITY_REPORT_METRICS],
                title="Failure and Judge reliability",
            )
            usage = partition.get("aggregate_score", {}).get("usage", {})
            lines.extend(["### Usage", ""])
            lines.append(
                "| Field | Sum | Mean | Observed | Population | Missing | Unit |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
            for field in (
                "elapsed_seconds",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "tool_calls",
                "cost",
            ):
                item = usage.get(field)
                if not isinstance(item, Mapping):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            _markdown_cell(field),
                            _markdown_cell(item.get("sum_value")),
                            _markdown_cell(item.get("mean_value")),
                            _markdown_cell(item.get("observed_count")),
                            _markdown_cell(item.get("population_count")),
                            _markdown_cell(item.get("missing_count")),
                            _markdown_cell(item.get("unit")),
                        )
                    )
                    + " |"
                )
            lines.append(
                f"- Cost currency: `{usage.get('cost_currency') or 'missing/unspecified'}`"
            )
            lines.append("")
            groupings = partition.get("groupings", [])
            if groupings:
                lines.extend(["### Groupings", ""])
                for grouping in groupings:
                    lines.append(
                        f"- `{canonical_json(grouping.get('dimension_names', []))}`: "
                        f"{len(grouping.get('scores', []))} aggregate(s)"
                    )
                lines.append("")

    diagnostics = payload["diagnostics"]
    lines.extend(["## Diagnostics", ""])
    _render_diagnostic_table(
        lines,
        "Critical/high required misses",
        diagnostics.get("critical_high_misses", []),
        ("task_id", "trial_id", "truth_id", "severity", "category", "source_status"),
    )
    _render_diagnostic_table(
        lines,
        "Fabricated Findings",
        diagnostics.get("fabricated_findings", []),
        ("task_id", "trial_id", "finding_id", "disposition", "evidence_integrity"),
    )
    _render_diagnostic_table(
        lines,
        "Novel Findings disallowed by policy",
        diagnostics.get("novel_disallowed_findings", []),
        ("task_id", "trial_id", "finding_id", "disposition", "issue_judgement"),
    )
    _render_diagnostic_table(
        lines,
        "Agent failures",
        diagnostics.get("agent_failures", []),
        ("task_id", "trial_id", "status", "failure_code", "retryable"),
    )
    _render_diagnostic_table(
        lines,
        "Ungraded evaluator outputs",
        diagnostics.get("ungraded_trials", []),
        ("task_id", "trial_id", "phase", "status", "result_digest"),
    )
    _render_diagnostic_table(
        lines,
        "Pending evaluator work",
        diagnostics.get("pending_evaluations", []),
        ("task_id", "trial_id", "phase", "status", "result_digest"),
    )
    _render_diagnostic_table(
        lines,
        "Missing evaluator outputs",
        diagnostics.get("missing_evaluations", []),
        ("task_id", "trial_id", "phase", "status"),
    )
    _render_diagnostic_table(
        lines,
        "Harness nonterminal Trials",
        diagnostics.get("nonterminal_trials", []),
        ("task_id", "trial_id", "status", "reason", "stages"),
    )
    _render_diagnostic_table(
        lines,
        "Judge failures",
        diagnostics.get("judge_failures", []),
        ("task_id", "trial_id", "phase", "request_id", "failure_code"),
    )
    _render_diagnostic_table(
        lines,
        "Judge ungraded",
        diagnostics.get("judge_ungraded", []),
        ("task_id", "trial_id", "phase", "request_id", "ungraded_reason"),
    )
    _render_diagnostic_table(
        lines,
        "Semantic unknown decisions",
        diagnostics.get("semantic_unknowns", []),
        ("task_id", "trial_id", "phase", "request_id", "task"),
    )
    lines.extend(["## Ground-truth scope", ""])
    lines.append(
        "Each partition retains its declared truth completeness; metrics from "
        "different completeness policies are not rolled into one value."
    )
    lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def render_trial_markdown(inspection: TrialInspection) -> str:
    """Render a metadata-only, deterministic view of one Trial inspection."""

    if type(inspection) is not TrialInspection:
        raise _error("render_trial_markdown requires TrialInspection")
    payload = inspection.to_dict()
    bindings = payload["source_bindings"]
    lines = [
        "# Code Review Trial Inspection",
        "",
        f"- Inspection ID: `{payload['inspection_id']}`",
        f"- Task: `{bindings.get('task_id')}`",
        f"- Trial: `{bindings.get('trial_id')}` (index `{bindings.get('trial_index')}`)",
        f"- Submission digest: `{bindings.get('submission_digest')}`",
        f"- Trial score digest: `{bindings.get('trial_score_digest')}`",
        "",
        "## Timeline",
        "",
    ]
    timeline = payload.get("timeline", [])
    if timeline:
        for receipt in timeline:
            lines.append(
                f"- `{receipt.get('stage')}` receipt `{receipt.get('receipt_id')}` "
                f"status: `{receipt.get('terminal_status')}`"
            )
    else:
        lines.append("No stage receipts were supplied in this projection.")
    lines.extend(["", "## Evaluations", ""])
    for label, key in (("Intent", "intent_evaluation"), ("Review", "review_evaluation")):
        evaluation = payload.get(key)
        if evaluation is None:
            lines.append(f"- {label}: missing")
            continue
        evaluation_payload = evaluation.get("payload", {})
        lines.append(
            f"- {label}: status: `{evaluation_payload.get('status')}`, "
            f"digest: `{evaluation.get('source_digest')}`"
        )
    lines.append("")
    lines.extend(["## Judge receipts", ""])
    judge_refs = payload.get("judge_artifact_refs", {})
    for phase in sorted(judge_refs):
        refs = judge_refs[phase]
        lines.append(
            f"- {phase}: requests: {len(refs.get('requests', []))}, "
            f"decisions: {len(refs.get('decisions', []))}, "
            f"failures: {len(refs.get('failures', []))}, "
            f"ungraded: {len(refs.get('ungraded', []))}"
        )
    lines.extend(["", "## Evidence diagnostics", ""])
    lines.append(f"- Finding Evidence records: {len(payload.get('evidence_diagnostics', []))}")
    clarification = payload.get("clarification_match_receipts")
    if clarification is None:
        lines.append("- Clarification receipts: none")
    else:
        lines.append(
            f"- Clarification receipts: {len(clarification.get('receipt_refs', []))}"
        )
    trace = payload.get("trace", {})
    trace_ref = trace.get("trace_ref")
    capture = trace.get("capture", {})
    lines.extend(
        [
            "",
            "## Trace metadata",
            "",
            f"- Trace ref: `{canonical_json(trace_ref) if trace_ref is not None else 'null'}`",
            f"- Captured bytes: `{capture.get('total_bytes')}`",
            f"- Captured files: `{len(capture.get('files', []))}`",
            "",
        ]
    )
    return "\n".join(lines).rstrip("\n") + "\n"


__all__ = [
    "RUN_REPORT_SUMMARY_SCHEMA_VERSION",
    "TRIAL_INSPECTION_SCHEMA_VERSION",
    "REPORT_REVISION",
    "REDACTED_ARTIFACT_PROJECTION_VERSION",
    "MAX_REPORT_BYTES",
    "MAX_INSPECTION_BYTES",
    "ReportError",
    "TrialEvaluationSource",
    "RunReportSummary",
    "TrialInspection",
    "ReportBuilder",
    "render_run_markdown",
    "render_trial_markdown",
]
