"""Black-box Agent Runner for the canonical code-review Eval harness.

The Runner is deliberately a coordinator, not an evaluator.  It only sees
the immutable ``EvalInput`` projection from a Run snapshot, prepares a clean
repository workspace, invokes one Adapter, validates the resulting
``EvalSubmission`` and commits the terminal artifact through ``ArtifactStore``.
Truth, metrics, Judge prompts and product internals never enter this module.

The public entry point supports both common workflows::

    runner.run(config, case_snapshot)       # preflight, create, then run
    runner.run(run_id)                       # resume an existing Run

Capability preflight is intentionally separate from Trial execution.  In
strict mode an incompatible Run is rejected before any Trial is started.  In
filter mode a new truth-free Case Snapshot and Run identity are created; the
incompatible Cases are not represented as Agent failures.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import os
import stat
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from .adapters.base import (
    AdapterCompatibility,
    AgentAdapterError,
    AgentAdapterIncompatibleError,
    AgentRunConfig,
    AgentUnderTestAdapter,
)
from .artifacts import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactStore,
    ExecutionArtifactBudgetError,
    RunManifest,
    RunStatus,
    StageName,
    TrialManifest,
    TrialState,
)
from .cases import RunCaseSnapshot
from .clarification import (
    BuiltinMaterialClaimMatcherFactory,
    ClarificationChannel,
    ClarificationMatcherError,
    ClarificationProtocolError,
    ClarificationSession,
    MaterialClaimMatchReceipt,
)
from .config import EvalRunConfig, SuiteRunConfig
from .models import (
    ClarificationScript,
    EvalInput,
    EvalSubmission,
    FailureCode,
    SchemaError,
    SubmissionStatus,
    TraceType,
    TrialStatus,
    canonical_json_bytes,
    canonical_sha256,
)
from .repository import (
    PreparedRepository,
    RepositoryPreparer,
    TrialWorkspace,
    WorkspaceManifest,
)
from .submission import (
    empty_usage,
    failure_submission,
    validate_submission_binding,
    validate_submission_trace,
)


RUNNER_SCHEMA_VERSION = "eval_runner_v1"
CAPABILITY_PREFLIGHT_SCHEMA_VERSION = "eval_capability_preflight_v1"
TERMINAL_SUMMARY_SCHEMA_VERSION = "eval_terminal_summary_v1"
CLARIFICATION_RECEIPT_SCHEMA_VERSION = "eval_clarification_match_receipts_v1"
TRACE_CAPTURE_SCHEMA_VERSION = "eval_trace_capture_v1"

MAX_PREFLIGHT_ISSUES = 100_000
MAX_PREFLIGHT_DETAIL_CHARS = 512
MAX_TERMINAL_SUMMARY_CHARS = 512
MAX_ADAPTER_DIAGNOSTIC_BYTES = 4 * 1024
TRACE_READ_CHUNK_BYTES = 64 * 1024
MAX_TRACE_NODES = 100_000
ADAPTER_IDENTITY_MISMATCH = "runner_incompatible.adapter_identity_mismatch"


class RunnerError(RuntimeError):
    """Base class for orchestration failures."""


class _AdapterIdentityMismatch(RunnerError):
    """A per-Trial Adapter instance drifted from the preflight identity."""


class _AdapterFactoryError(RunnerError):
    """The Harness could not construct a per-Trial Adapter instance."""


class RunIncompatibilityError(RunnerError):
    """The Adapter cannot represent one or more Cases in the requested Run."""

    def __init__(
        self,
        message: str,
        *,
        preflight: "CapabilityPreflight",
        config: EvalRunConfig,
        snapshot: RunCaseSnapshot,
    ) -> None:
        super().__init__(message)
        self.preflight = preflight
        self.config = config
        self.snapshot = snapshot


class CapabilityPolicy(str, Enum):
    """How preflight incompatibilities affect a Run."""

    STRICT = "strict"
    FILTER = "filter"


# A short alias is useful to callers that describe this as a preflight mode.
PreflightMode = CapabilityPolicy


@dataclass(frozen=True)
class CapabilityIssue:
    task_id: str
    trial_index: int
    reason: str
    unsupported: Tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise SchemaError("preflight issue.task_id must be a non-empty string")
        if type(self.trial_index) is not int or self.trial_index < 1:
            raise SchemaError("preflight issue.trial_index must be positive")
        if not isinstance(self.reason, str) or not self.reason:
            raise SchemaError("preflight issue.reason must be a non-empty string")
        if len(self.reason) > MAX_PREFLIGHT_DETAIL_CHARS:
            raise SchemaError("preflight issue.reason is too long")
        if type(self.unsupported) is not tuple or any(
            not isinstance(item, str) or not item for item in self.unsupported
        ):
            raise SchemaError("preflight issue.unsupported must contain strings")
        if len(self.unsupported) != len(set(self.unsupported)):
            raise SchemaError("preflight issue.unsupported contains duplicates")
        if not isinstance(self.detail, str):
            raise SchemaError("preflight issue.detail must be a string")
        if len(self.detail) > MAX_PREFLIGHT_DETAIL_CHARS:
            raise SchemaError("preflight issue.detail is too long")

    @classmethod
    def from_dict(cls, value: Any) -> "CapabilityIssue":
        if type(value) is not dict:
            raise SchemaError("preflight issue must be an object")
        expected = {"task_id", "trial_index", "reason", "unsupported", "detail"}
        if set(value) != expected:
            raise SchemaError("preflight issue has unexpected fields")
        if type(value["unsupported"]) is not list:
            raise SchemaError("preflight issue.unsupported must be a list")
        return cls(
            task_id=value["task_id"],
            trial_index=value["trial_index"],
            reason=value["reason"],
            unsupported=tuple(value["unsupported"]),
            detail=value["detail"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trial_index": self.trial_index,
            "reason": self.reason,
            "unsupported": list(self.unsupported),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CapabilityPreflight:
    """Immutable, truth-free capability coverage for one candidate Run."""

    run_id: str
    agent_id: str
    adapter_id: str
    adapter_version: str
    policy: CapabilityPolicy
    checked_trials: Tuple[Tuple[str, int], ...]
    compatible_task_ids: Tuple[str, ...]
    incompatible_task_ids: Tuple[str, ...]
    issues: Tuple[CapabilityIssue, ...]
    filtered_from_run_id: Optional[str] = None

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("run_id", self.run_id, 512),
            ("agent_id", self.agent_id, 512),
            ("adapter_id", self.adapter_id, 512),
            ("adapter_version", self.adapter_version, 256),
        ):
            if not isinstance(value, str) or not value or len(value) > maximum:
                raise SchemaError("preflight.%s is invalid" % name)
        if not isinstance(self.policy, CapabilityPolicy):
            raise TypeError("preflight.policy must be a CapabilityPolicy")
        if type(self.checked_trials) is not tuple:
            raise SchemaError("preflight.checked_trials must be a tuple")
        for item in self.checked_trials:
            if (
                type(item) is not tuple
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or type(item[1]) is not int
                or item[1] < 1
            ):
                raise SchemaError("preflight.checked_trials contains an invalid binding")
        if len(self.checked_trials) != len(set(self.checked_trials)):
            raise SchemaError("preflight.checked_trials contains duplicates")
        for name, values in (
            ("compatible_task_ids", self.compatible_task_ids),
            ("incompatible_task_ids", self.incompatible_task_ids),
        ):
            if type(values) is not tuple or any(
                not isinstance(item, str) or not item for item in values
            ):
                raise SchemaError("preflight.%s must contain strings" % name)
            if len(values) != len(set(values)):
                raise SchemaError("preflight.%s contains duplicates" % name)
        if type(self.issues) is not tuple or any(
            not isinstance(item, CapabilityIssue) for item in self.issues
        ):
            raise SchemaError("preflight.issues must contain CapabilityIssue values")
        if len(self.issues) > MAX_PREFLIGHT_ISSUES:
            raise ValueError("preflight contains too many issues")
        if set(self.compatible_task_ids).intersection(self.incompatible_task_ids):
            raise ValueError("preflight task coverage overlaps")
        checked_task_ids = {item[0] for item in self.checked_trials}
        covered_task_ids = set(self.compatible_task_ids).union(
            self.incompatible_task_ids
        )
        if checked_task_ids != covered_task_ids:
            raise SchemaError("preflight task coverage does not match checked Trials")
        if self.filtered_from_run_id is not None and (
            not isinstance(self.filtered_from_run_id, str)
            or not self.filtered_from_run_id
            or len(self.filtered_from_run_id) > 512
        ):
            raise SchemaError("preflight.filtered_from_run_id is invalid")
        if self.filtered_from_run_id == self.run_id:
            raise ValueError("filtered preflight cannot point at itself")

    @property
    def compatible(self) -> bool:
        return not self.incompatible_task_ids

    @property
    def coverage(self) -> Dict[str, int]:
        return {
            "checked_trials": len(self.checked_trials),
            "compatible_cases": len(self.compatible_task_ids),
            "incompatible_cases": len(self.incompatible_task_ids),
            "issues": len(self.issues),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": CAPABILITY_PREFLIGHT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "policy": self.policy.value,
            "checked_trials": [
                {"task_id": task_id, "trial_index": trial_index}
                for task_id, trial_index in self.checked_trials
            ],
            "compatible_task_ids": list(self.compatible_task_ids),
            "incompatible_task_ids": list(self.incompatible_task_ids),
            "issues": [item.to_dict() for item in self.issues],
            "coverage": self.coverage,
            "filtered_from_run_id": self.filtered_from_run_id,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "CapabilityPreflight":
        if type(value) is not dict:
            raise SchemaError("capability preflight must be an object")
        expected = {
            "schema_version",
            "run_id",
            "agent_id",
            "adapter_id",
            "adapter_version",
            "policy",
            "checked_trials",
            "compatible_task_ids",
            "incompatible_task_ids",
            "issues",
            "coverage",
            "filtered_from_run_id",
        }
        if set(value) != expected:
            raise SchemaError("capability preflight has unexpected fields")
        if value["schema_version"] != CAPABILITY_PREFLIGHT_SCHEMA_VERSION:
            raise SchemaError("capability preflight has an unknown schema_version")
        for name in (
            "checked_trials",
            "compatible_task_ids",
            "incompatible_task_ids",
            "issues",
        ):
            if type(value[name]) is not list:
                raise SchemaError("preflight.%s must be a list" % name)
        if len(value["issues"]) > MAX_PREFLIGHT_ISSUES:
            raise SchemaError("preflight contains too many issues")
        checked: List[Tuple[str, int]] = []
        for item in value["checked_trials"]:
            if type(item) is not dict or set(item) != {"task_id", "trial_index"}:
                raise SchemaError("preflight checked Trial has unexpected fields")
            checked.append((item["task_id"], item["trial_index"]))
        try:
            policy = CapabilityPolicy(value["policy"])
        except (TypeError, ValueError) as exc:
            raise SchemaError("preflight.policy is invalid") from exc
        result = cls(
            run_id=value["run_id"],
            agent_id=value["agent_id"],
            adapter_id=value["adapter_id"],
            adapter_version=value["adapter_version"],
            policy=policy,
            checked_trials=tuple(checked),
            compatible_task_ids=tuple(value["compatible_task_ids"]),
            incompatible_task_ids=tuple(value["incompatible_task_ids"]),
            issues=tuple(CapabilityIssue.from_dict(item) for item in value["issues"]),
            filtered_from_run_id=value["filtered_from_run_id"],
        )
        if type(value["coverage"]) is not dict or value["coverage"] != result.coverage:
            raise SchemaError("preflight.coverage does not match its payload")
        return result

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class RunSetup:
    """The immutable objects produced after preflight and Run creation."""

    config: EvalRunConfig
    case_snapshot: RunCaseSnapshot
    manifest: RunManifest
    preflight: CapabilityPreflight


@dataclass(frozen=True)
class TrialResult:
    run_id: str
    task_id: str
    trial_id: str
    trial_index: int
    status: TrialStatus
    submission: Optional[EvalSubmission]
    attempt: Optional[int]
    skipped: bool
    workspace_binding_id: Optional[str]
    incompatibility: Optional[str]
    diagnostic: str

    @property
    def terminal(self) -> bool:
        return self.status in {
            TrialStatus.COMPLETED,
            TrialStatus.FAILED,
            TrialStatus.BLOCKED,
            TrialStatus.INVALID_OUTPUT,
        }


@dataclass(frozen=True)
class RunResult:
    run_id: str
    config: EvalRunConfig
    preflight: CapabilityPreflight
    trials: Tuple[TrialResult, ...]
    status: RunStatus
    created: bool

    @property
    def submissions(self) -> Tuple[EvalSubmission, ...]:
        return tuple(
            item.submission for item in self.trials if item.submission is not None
        )

    @property
    def terminal(self) -> bool:
        return self.status is RunStatus.COMPLETED


@runtime_checkable
class ClarificationScriptProvider(Protocol):
    """Minimal Runner-side provider; it exposes no truth to the Adapter."""

    def clarification_script(self, task_id: str) -> ClarificationScript:
        ...


@dataclass(frozen=True)
class AdapterDiagnostic:
    """Optional bounded diagnostic data exposed by a trusted Adapter."""

    stdout: bytes = b""
    stderr: bytes = b""
    stdout_bytes: Optional[int] = None
    stderr_bytes: Optional[int] = None


class _LazyClarificationSession:
    """Do not instantiate a matcher until the Adapter actually asks."""

    def __init__(
        self,
        *,
        task_id: str,
        provider: Any,
        binding: AgentRunConfig,
        matcher_factory: Any,
    ) -> None:
        self._task_id = task_id
        self._provider = provider
        self._binding = binding
        self._matcher_factory = matcher_factory
        self._session: Optional[ClarificationSession] = None

    def _ensure(self) -> ClarificationSession:
        if self._session is None:
            try:
                script = _load_clarification_script(self._provider, self._task_id)
                self._session = ClarificationSession(
                    script,
                    run_binding=self._binding,
                    matcher_factory=self._matcher_factory,
                )
            except (RunnerError, ClarificationProtocolError):
                raise
            except Exception as exc:
                raise RunnerError("clarification matcher setup failed") from exc
        return self._session

    @property
    def channel(self) -> ClarificationChannel:
        return _LazyClarificationChannel(self)

    @property
    def transcript(self) -> Tuple[Any, ...]:
        return () if self._session is None else self._session.transcript

    @property
    def match_receipts(self) -> Tuple[MaterialClaimMatchReceipt, ...]:
        return () if self._session is None else self._session.match_receipts

    def ask(self, **kwargs: Any) -> Any:
        try:
            return self._ensure().channel.ask(**kwargs)
        except ClarificationMatcherError as exc:
            raise RunnerError("clarification matcher execution failed") from exc
        except (RunnerError, ClarificationProtocolError):
            raise
        except Exception as exc:
            raise RunnerError("clarification matcher execution failed") from exc


class _LazyClarificationChannel:
    def __init__(self, owner: _LazyClarificationSession) -> None:
        self._owner = owner

    def ask(self, **kwargs: Any) -> Any:
        return self._owner.ask(**kwargs)


class _CombinedCancelEvent:
    """Expose one cooperative signal backed by Runner and caller events."""

    def __init__(self, *events: Any) -> None:
        self._events = tuple(event for event in events if event is not None)

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)

    def wait(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                time.sleep(min(0.02, remaining))
            else:
                time.sleep(0.02)
        return True


def _load_clarification_script(provider: Any, task_id: str) -> ClarificationScript:
    """Load only the script projection from a Runner-private provider."""

    if provider is None:
        raise RunnerError("clarification script provider is required")
    value: Any
    if isinstance(provider, Mapping):
        if task_id not in provider:
            raise RunnerError("clarification script is not bound to the Case")
        value = provider[task_id]
    elif hasattr(provider, "clarification_script"):
        value = provider.clarification_script(task_id)
    elif hasattr(provider, "runner_case"):
        value = provider.runner_case(task_id)
    elif hasattr(provider, "evaluator_case"):
        value = provider.evaluator_case(task_id)
    elif callable(provider):
        value = provider(task_id)
    else:
        raise RunnerError("unsupported clarification script provider")
    if isinstance(value, ClarificationScript):
        return value
    # EvalCase is intentionally duck-read only for this one field.  The
    # Runner never accesses intent_truth or review_truth.
    script = getattr(value, "clarification_script", None)
    if isinstance(script, ClarificationScript):
        return script
    raise RunnerError("clarification provider did not return a ClarificationScript")


def _adapter_identity(adapter: Any) -> Tuple[str, str]:
    adapter_id = getattr(adapter, "ADAPTER_KIND", None)
    if not isinstance(adapter_id, str) or not adapter_id:
        adapter_id = getattr(adapter, "KIND", None)
    if not isinstance(adapter_id, str) or not adapter_id:
        adapter_id = "%s.%s" % (
            adapter.__class__.__module__,
            adapter.__class__.__qualname__,
        )
    adapter_version = getattr(adapter, "ADAPTER_VERSION", None)
    if not isinstance(adapter_version, str) or not adapter_version:
        adapter_version = getattr(adapter, "VERSION", None)
    if not isinstance(adapter_version, str) or not adapter_version:
        adapter_version = "1"
    if len(adapter_id) > 512 or len(adapter_version) > 256:
        raise RunnerError("Adapter identity exceeds its bounded length")
    return adapter_id, adapter_version


def _adapter_supports_cancellation(adapter: Any) -> bool:
    """Detect the optional keyword without probing the Adapter twice."""

    try:
        parameters = inspect.signature(adapter.run).parameters.values()
    except (TypeError, ValueError, AttributeError):
        return False
    return any(
        parameter.name == "cancel_event"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _invoke_adapter(
    adapter: AgentUnderTestAdapter,
    eval_input: EvalInput,
    workspace: Path,
    config: AgentRunConfig,
    clarification_channel: ClarificationChannel,
    cancel_event: Any,
    *,
    target_materialization_id: str,
) -> EvalSubmission:
    if _adapter_supports_cancellation(adapter):
        return adapter.run(
            eval_input,
            workspace,
            config,
            clarification_channel,
            target_materialization_id=target_materialization_id,
            cancel_event=cancel_event,
        )
    return adapter.run(
        eval_input,
        workspace,
        config,
        clarification_channel,
        target_materialization_id=target_materialization_id,
    )


def _failure_message(code: FailureCode) -> str:
    return {
        FailureCode.TIMEOUT: "Agent execution exceeded its time limit",
        FailureCode.NON_ZERO_EXIT: "Agent execution exited unsuccessfully",
        FailureCode.PROCESS_KILLED: "Agent execution was interrupted or killed",
        FailureCode.OUTPUT_OVERFLOW: "Agent output exceeded its configured limit",
        FailureCode.INVALID_JSON: "Agent output was not valid JSON",
        FailureCode.SCHEMA_MISMATCH: "Agent output did not match EvalSubmission v2",
        FailureCode.CLARIFICATION_REQUIRED: "Agent requires clarification",
        FailureCode.AGENT_BLOCKED: "Agent reported a blocked execution",
        FailureCode.ADAPTER_ERROR: "Agent Adapter failed at its execution boundary",
        FailureCode.UNKNOWN: "Agent ended without a canonical terminal result",
    }[code]


def _exception_failure_code(exc: BaseException) -> FailureCode:
    if isinstance(exc, KeyboardInterrupt):
        return FailureCode.PROCESS_KILLED
    if isinstance(exc, (TimeoutError,)):
        return FailureCode.TIMEOUT
    if isinstance(exc, (ProcessLookupError, BrokenPipeError)):
        return FailureCode.PROCESS_KILLED
    return FailureCode.ADAPTER_ERROR


def _terminal_status_for_submission(submission: EvalSubmission) -> TrialStatus:
    return TrialStatus(submission.status.value)


def _safe_diag_text(value: Any) -> str:
    text = str(value)
    if len(text) > MAX_TERMINAL_SUMMARY_CHARS:
        text = text[:MAX_TERMINAL_SUMMARY_CHARS]
    # Do not persist exception messages, argv, or output excerpts.  The type
    # and stable bounded code are enough for a report; raw diagnostics remain
    # outside the control plane.
    return text.replace("\r", " ").replace("\n", " ")


def _stream_summary(data: bytes, declared: Optional[int]) -> Dict[str, Any]:
    if type(data) is not bytes:
        data = b""
    bounded = data[:MAX_ADAPTER_DIAGNOSTIC_BYTES]
    total = (
        declared
        if type(declared) is int and declared >= len(data)
        else len(data)
    )
    return {
        "bytes": total,
        "sha256": hashlib.sha256(bounded).hexdigest(),
        "sha256_scope": "captured_prefix",
        "summary_bytes": len(bounded),
        "truncated": total > len(bounded),
    }


def _read_adapter_diagnostic(adapter: Any) -> AdapterDiagnostic:
    try:
        value = getattr(adapter, "last_diagnostics", None)
    except Exception:
        return AdapterDiagnostic()
    if callable(value):
        try:
            value = value()
        except Exception:
            value = None
    if isinstance(value, AdapterDiagnostic):
        return AdapterDiagnostic(
            stdout=(
                value.stdout[:MAX_ADAPTER_DIAGNOSTIC_BYTES]
                if type(value.stdout) is bytes
                else b""
            ),
            stderr=(
                value.stderr[:MAX_ADAPTER_DIAGNOSTIC_BYTES]
                if type(value.stderr) is bytes
                else b""
            ),
            stdout_bytes=(
                value.stdout_bytes
                if type(value.stdout_bytes) is int and value.stdout_bytes >= 0
                else len(value.stdout)
                if type(value.stdout) is bytes
                else 0
            ),
            stderr_bytes=(
                value.stderr_bytes
                if type(value.stderr_bytes) is int and value.stderr_bytes >= 0
                else len(value.stderr)
                if type(value.stderr) is bytes
                else 0
            ),
        )
    if isinstance(value, Mapping):
        stdout = value.get("stdout", b"")
        stderr = value.get("stderr", b"")
        return AdapterDiagnostic(
            stdout=(
                stdout[:MAX_ADAPTER_DIAGNOSTIC_BYTES]
                if type(stdout) is bytes
                else b""
            ),
            stderr=(
                stderr[:MAX_ADAPTER_DIAGNOSTIC_BYTES]
                if type(stderr) is bytes
                else b""
            ),
            stdout_bytes=(
                value.get("stdout_bytes")
                if type(value.get("stdout_bytes")) is int
                and value.get("stdout_bytes") >= 0
                else len(stdout)
                if type(stdout) is bytes
                else 0
            ),
            stderr_bytes=(
                value.get("stderr_bytes")
                if type(value.get("stderr_bytes")) is int
                and value.get("stderr_bytes") >= 0
                else len(stderr)
                if type(stderr) is bytes
                else 0
            ),
        )
    return AdapterDiagnostic()


def _match_receipt_dict(receipt: MaterialClaimMatchReceipt) -> Dict[str, Any]:
    return {
        "turn_index": receipt.turn_index,
        "question_id": receipt.question_id,
        "dimension": receipt.dimension.value,
        "actual_claim_digest": receipt.actual_claim_digest,
        "matcher_digest": receipt.matcher_digest,
        "candidates": [
            {
                "answer_id": item.answer_id,
                "request_digest": item.request_digest,
                "equivalent": item.equivalent,
                "action_eligible": item.action_eligible,
            }
            for item in receipt.candidates
        ],
        "outcome": receipt.outcome.value,
        "matched_answer_id": receipt.matched_answer_id,
    }


def _clarification_artifact(
    binding: AgentRunConfig,
    controller: _LazyClarificationSession,
) -> Dict[str, Any]:
    return {
        "schema_version": CLARIFICATION_RECEIPT_SCHEMA_VERSION,
        "trial_id": binding.trial_id,
        "matcher_digest": binding.clarification_matcher_config_digest,
        "receipts": [_match_receipt_dict(item) for item in controller.match_receipts],
    }


def _workspace_manifest_value(handle: Any) -> Optional[Dict[str, Any]]:
    manifest = getattr(handle, "manifest", None)
    if not isinstance(manifest, WorkspaceManifest):
        return None
    return manifest.to_dict()


def _capture_trace_summary(
    submission: EvalSubmission,
    workspace: Optional[Path],
    *,
    max_trace_bytes: int,
) -> Optional[Dict[str, Any]]:
    """Capture a bounded Harness-private trace before workspace cleanup.

    The canonical Submission keeps the Agent's original ``TraceRef``.  The
    private execution artifact preserves exact bytes (base64 in canonical
    JSON), hash and size so a cleaned workspace does not leave an unexplained
    dangling reference.  User-facing reports must expose only its summary.
    """

    trace_ref = submission.trace_ref
    if trace_ref is None:
        return None
    if workspace is None or trace_ref.type is not TraceType.LOCAL_PATH:
        return {
            "schema_version": TRACE_CAPTURE_SCHEMA_VERSION,
            "captured": False,
            "reason": "non_local_or_unavailable_trace",
            "trace_ref": trace_ref.to_dict(),
            "files": [],
            "total_bytes": None,
        }
    try:
        if type(max_trace_bytes) is not int or max_trace_bytes < 1:
            raise ValueError("max_trace_bytes must be a positive integer")

        candidate = Path(trace_ref.value)
        trace_parts = tuple(
            part for part in candidate.parts if part not in {"", "."}
        )
        if (
            candidate.is_absolute()
            or bool(candidate.drive)
            or not trace_parts
            or any(part == ".." for part in trace_parts)
        ):
            raise ValueError("trace path must be a safe relative path")

        files: List[Dict[str, Any]] = []
        total = 0
        nodes = 0

        def identity(info: os.stat_result) -> Tuple[Any, ...]:
            object_identity = (
                getattr(info, "st_dev", None),
                getattr(info, "st_ino", None),
                stat.S_IFMT(info.st_mode),
            )
            if stat.S_ISDIR(info.st_mode):
                # Ancestor directories may legitimately gain unrelated
                # siblings (for example from another parallel Trial).  Their
                # directory-entry binding is the security invariant; mutable
                # directory timestamps and link counts are not.
                return object_identity
            return object_identity + (
                getattr(info, "st_nlink", None),
                getattr(info, "st_size", None),
                getattr(info, "st_mtime_ns", None),
                getattr(info, "st_ctime_ns", None),
            )

        def _trace_unsafe_node(info: os.stat_result) -> bool:
            attributes = getattr(info, "st_file_attributes", 0)
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            return bool(
                stat.S_ISLNK(info.st_mode)
                or attributes & reparse
                or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1)
            )

        def _assert_safe_node(
            info: os.stat_result,
            *,
            expected_kind: Optional[str] = None,
        ) -> str:
            if _trace_unsafe_node(info):
                raise ValueError("trace contains an unsafe filesystem node")
            if stat.S_ISDIR(info.st_mode):
                kind = "directory"
            elif stat.S_ISREG(info.st_mode):
                kind = "file"
            else:
                raise ValueError("trace contains a special file")
            if expected_kind is not None and kind != expected_kind:
                raise ValueError("trace node type changed during capture")
            return kind

        def _append_file(
            relative: str,
            data: bytes,
            digest: str,
            size: int,
        ) -> None:
            nonlocal total
            total += size
            files.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "sha256": digest,
                    "content_base64": base64.b64encode(data).decode("ascii"),
                    "content_truncated": False,
                }
            )

        def _read_descriptor(
            descriptor: int,
            expected: os.stat_result,
        ) -> Tuple[bytes, str, int]:
            if expected.st_size > max_trace_bytes - total:
                raise ValueError("trace exceeds its configured byte limit")
            opened = os.fstat(descriptor)
            _assert_safe_node(opened, expected_kind="file")
            if identity(opened) != identity(expected):
                raise ValueError("trace file changed during safe open")
            captured = bytearray()
            digest = hashlib.sha256()
            read_total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(
                        TRACE_READ_CHUNK_BYTES,
                        max(1, max_trace_bytes - total - read_total + 1),
                    ),
                )
                if not chunk:
                    break
                read_total += len(chunk)
                if total + read_total > max_trace_bytes:
                    raise ValueError("trace exceeds its configured byte limit")
                digest.update(chunk)
                captured.extend(chunk)
            after = os.fstat(descriptor)
            _assert_safe_node(after, expected_kind="file")
            if identity(after) != identity(expected):
                raise ValueError("trace file changed while being captured")
            if read_total != expected.st_size:
                raise ValueError("trace file size drifted while being captured")
            return bytes(captured), digest.hexdigest(), read_total

        def _capture_posix() -> None:
            required_dir_fd = (
                os.open in os.supports_dir_fd
                and os.stat in os.supports_dir_fd
                and os.stat in os.supports_follow_symlinks
                and os.scandir in os.supports_fd
                and hasattr(os, "O_NOFOLLOW")
            )
            if not required_dir_fd:
                raise OSError("secure fd-relative trace capture is unavailable")

            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | os.O_NOFOLLOW
            )
            child_flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | os.O_NOFOLLOW
            )

            # A frame keeps both ends of a directory entry binding alive.  The
            # parent descriptor makes later verification independent of path
            # replacement higher in the tree.
            frames: List[Tuple[int, str, int, os.stat_result]] = []
            persistent_fds: List[int] = []
            root_fd = os.open("/", directory_flags)
            persistent_fds.append(root_fd)
            filesystem_root = os.fstat(root_fd)
            _assert_safe_node(filesystem_root, expected_kind="directory")

            def _open_child(
                parent_fd: int,
                name: str,
                *,
                expected_kind: Optional[str] = None,
            ) -> Tuple[int, os.stat_result, str]:
                before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                kind = _assert_safe_node(before, expected_kind=expected_kind)
                flags = child_flags
                if kind == "directory":
                    flags |= getattr(os, "O_DIRECTORY", 0)
                descriptor = os.open(name, flags, dir_fd=parent_fd)
                try:
                    opened = os.fstat(descriptor)
                    _assert_safe_node(opened, expected_kind=kind)
                    if identity(opened) != identity(before):
                        raise ValueError("trace node changed during safe open")
                    rebound = os.stat(
                        name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    _assert_safe_node(rebound, expected_kind=kind)
                    if identity(rebound) != identity(opened):
                        raise ValueError("trace node binding changed during open")
                    return descriptor, opened, kind
                except BaseException:
                    os.close(descriptor)
                    raise

            def _verify_frame(
                frame: Tuple[int, str, int, os.stat_result]
            ) -> None:
                parent_fd, name, descriptor, expected = frame
                kind = _assert_safe_node(expected)
                held = os.fstat(descriptor)
                _assert_safe_node(held, expected_kind=kind)
                if identity(held) != identity(expected):
                    raise ValueError("trace node identity drifted")
                rebound = os.stat(
                    name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                _assert_safe_node(rebound, expected_kind=kind)
                if identity(rebound) != identity(expected):
                    raise ValueError("trace parent binding drifted")

            def _verify_chain() -> None:
                root_after = os.fstat(root_fd)
                _assert_safe_node(root_after, expected_kind="directory")
                if identity(root_after) != identity(filesystem_root):
                    raise ValueError("filesystem root identity drifted")
                for frame in frames:
                    _verify_frame(frame)

            def _directory_names(descriptor: int) -> Tuple[str, ...]:
                with os.scandir(descriptor) as entries:
                    names = tuple(sorted(entry.name for entry in entries))
                if any(
                    not name
                    or name in {".", ".."}
                    or "/" in name
                    or "\x00" in name
                    for name in names
                ):
                    raise ValueError("trace directory contains an unsafe name")
                return names

            def _walk(
                descriptor: int,
                info: os.stat_result,
                kind: str,
                relative_parts: Tuple[str, ...],
            ) -> None:
                nonlocal nodes
                nodes += 1
                if nodes > MAX_TRACE_NODES:
                    raise ValueError("trace exceeds its node limit")
                _verify_chain()
                relative = "/".join(relative_parts)
                if kind == "file":
                    data, digest, size = _read_descriptor(descriptor, info)
                    _verify_chain()
                    _append_file(relative, data, digest, size)
                    return

                names = _directory_names(descriptor)
                _verify_chain()
                for name in names:
                    child_fd, child_info, child_kind = _open_child(
                        descriptor,
                        name,
                    )
                    child_frame = (descriptor, name, child_fd, child_info)
                    frames.append(child_frame)
                    try:
                        _walk(
                            child_fd,
                            child_info,
                            child_kind,
                            relative_parts + (name,),
                        )
                        _verify_frame(child_frame)
                    finally:
                        frames.pop()
                        os.close(child_fd)
                    _verify_chain()
                if _directory_names(descriptor) != names:
                    raise ValueError("trace directory contents drifted")
                _verify_chain()

            try:
                workspace_path = Path(os.path.abspath(os.fspath(workspace)))
                if not workspace_path.is_absolute() or workspace_path.anchor != "/":
                    raise ValueError("workspace path is not a POSIX absolute path")
                current_fd = root_fd
                for part in workspace_path.parts[1:]:
                    descriptor, opened, kind = _open_child(
                        current_fd,
                        part,
                        expected_kind="directory",
                    )
                    if kind != "directory":
                        raise ValueError("workspace parent is not a directory")
                    frame = (current_fd, part, descriptor, opened)
                    frames.append(frame)
                    persistent_fds.append(descriptor)
                    current_fd = descriptor

                for index, part in enumerate(trace_parts):
                    is_last = index == len(trace_parts) - 1
                    descriptor, opened, kind = _open_child(
                        current_fd,
                        part,
                        expected_kind=None if is_last else "directory",
                    )
                    frame = (current_fd, part, descriptor, opened)
                    frames.append(frame)
                    persistent_fds.append(descriptor)
                    current_fd = descriptor

                trace_info = frames[-1][3]
                trace_kind = _assert_safe_node(trace_info)
                _walk(current_fd, trace_info, trace_kind, trace_parts)
                _verify_chain()
            finally:
                for descriptor in reversed(persistent_fds):
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass

        def _capture_windows() -> None:
            # Python does not expose openat/scandirat on Windows.  Keep stable
            # Win32 handles for every ancestor, reject reparse points, and
            # verify both file IDs and normalized final paths before and after
            # every path-based enumeration/read.  Bytes are committed only
            # after the complete handle chain remains unchanged.
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

            class _ByHandleFileInformation(ctypes.Structure):
                _fields_ = [
                    ("dwFileAttributes", wintypes.DWORD),
                    ("ftCreationTime", wintypes.FILETIME),
                    ("ftLastAccessTime", wintypes.FILETIME),
                    ("ftLastWriteTime", wintypes.FILETIME),
                    ("dwVolumeSerialNumber", wintypes.DWORD),
                    ("nFileSizeHigh", wintypes.DWORD),
                    ("nFileSizeLow", wintypes.DWORD),
                    ("nNumberOfLinks", wintypes.DWORD),
                    ("nFileIndexHigh", wintypes.DWORD),
                    ("nFileIndexLow", wintypes.DWORD),
                ]

            class _FileBasicInformation(ctypes.Structure):
                _fields_ = [
                    ("CreationTime", ctypes.c_longlong),
                    ("LastAccessTime", ctypes.c_longlong),
                    ("LastWriteTime", ctypes.c_longlong),
                    ("ChangeTime", ctypes.c_longlong),
                    ("FileAttributes", wintypes.DWORD),
                ]

            create_file = kernel32.CreateFileW
            create_file.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            create_file.restype = wintypes.HANDLE
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            get_information = kernel32.GetFileInformationByHandle
            get_information.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_ByHandleFileInformation),
            ]
            get_information.restype = wintypes.BOOL
            get_information_ex = kernel32.GetFileInformationByHandleEx
            get_information_ex.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                wintypes.LPVOID,
                wintypes.DWORD,
            ]
            get_information_ex.restype = wintypes.BOOL
            get_final_path = kernel32.GetFinalPathNameByHandleW
            get_final_path.argtypes = [
                wintypes.HANDLE,
                wintypes.LPWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
            ]
            get_final_path.restype = wintypes.DWORD
            read_file = kernel32.ReadFile
            read_file.argtypes = [
                wintypes.HANDLE,
                wintypes.LPVOID,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                wintypes.LPVOID,
            ]
            read_file.restype = wintypes.BOOL

            invalid_handle = ctypes.c_void_p(-1).value
            generic_read = 0x80000000
            file_list_directory = 0x0001
            file_read_attributes = 0x0080
            share_all = 0x00000001 | 0x00000002 | 0x00000004
            open_existing = 3
            flag_backup_semantics = 0x02000000
            flag_open_reparse_point = 0x00200000
            flag_sequential_scan = 0x08000000
            attribute_directory = 0x0010
            attribute_reparse_point = 0x0400
            file_basic_info_class = 0

            def _api_path(path: Path) -> str:
                value = os.path.abspath(os.fspath(path))
                if value.startswith("\\\\?\\"):
                    return value
                if value.startswith("\\\\"):
                    return "\\\\?\\UNC\\" + value[2:]
                return "\\\\?\\" + value

            def _raise_last_error(message: str) -> None:
                error = ctypes.get_last_error()
                raise OSError(error, message)

            def _win_info(handle: int) -> Tuple[Any, ...]:
                standard = _ByHandleFileInformation()
                if not get_information(handle, ctypes.byref(standard)):
                    _raise_last_error("could not query trace handle identity")
                basic = _FileBasicInformation()
                if not get_information_ex(
                    handle,
                    file_basic_info_class,
                    ctypes.byref(basic),
                    ctypes.sizeof(basic),
                ):
                    _raise_last_error("could not query trace handle change time")
                file_index = (
                    int(standard.nFileIndexHigh) << 32
                ) | int(standard.nFileIndexLow)
                size = (
                    int(standard.nFileSizeHigh) << 32
                ) | int(standard.nFileSizeLow)
                return (
                    int(standard.dwVolumeSerialNumber),
                    file_index,
                    int(standard.dwFileAttributes),
                    int(standard.nNumberOfLinks),
                    size,
                    int(basic.CreationTime),
                    int(basic.LastWriteTime),
                    int(basic.ChangeTime),
                )

            def _win_kind(info: Tuple[Any, ...]) -> str:
                if info[2] & attribute_reparse_point:
                    raise ValueError("trace contains a reparse point")
                kind = "directory" if info[2] & attribute_directory else "file"
                if kind == "file" and info[3] != 1:
                    raise ValueError("trace contains a hard-linked file")
                return kind

            def _same_win_object(
                left: Tuple[Any, ...],
                right: Tuple[Any, ...],
                kind: str,
            ) -> bool:
                # Volume + file ID anchors a directory even if parallel work
                # mutates unrelated entries beneath an ancestor.  Files also
                # bind link count, size, write time and change time so content
                # drift cannot pass as the same capture.
                if left[:2] != right[:2]:
                    return False
                left_type = left[2] & (
                    attribute_directory | attribute_reparse_point
                )
                right_type = right[2] & (
                    attribute_directory | attribute_reparse_point
                )
                if left_type != right_type:
                    return False
                if kind == "file":
                    return left[3:] == right[3:]
                return True

            def _normalized_final_path(handle: int) -> str:
                required = get_final_path(handle, None, 0, 0)
                if required == 0:
                    _raise_last_error("could not resolve trace handle path")
                buffer = ctypes.create_unicode_buffer(required + 1)
                written = get_final_path(handle, buffer, len(buffer), 0)
                if written == 0 or written >= len(buffer):
                    _raise_last_error("could not resolve complete trace handle path")
                return os.path.normcase(os.path.normpath(buffer.value))

            def _within(child: str, parent: str) -> bool:
                try:
                    common = os.path.commonpath((child, parent))
                except ValueError:
                    return False
                return os.path.normcase(common) == os.path.normcase(parent)

            def _stat_matches_handle(
                expected: os.stat_result,
                info: Tuple[Any, ...],
                kind: str,
            ) -> bool:
                expected_kind = _assert_safe_node(expected)
                if expected_kind != kind:
                    return False
                inode = getattr(expected, "st_ino", 0)
                if inode and int(inode) != info[1]:
                    return False
                if kind == "file":
                    return (
                        int(expected.st_nlink) == info[3]
                        and int(expected.st_size) == info[4]
                    )
                return True

            def _open_path(
                path: Path,
                *,
                expected_kind: str,
                for_read: bool = False,
            ) -> Tuple[int, Tuple[Any, ...], str]:
                before = os.lstat(path)
                _assert_safe_node(before, expected_kind=expected_kind)
                access = file_read_attributes
                flags = flag_open_reparse_point
                if expected_kind == "directory":
                    access |= file_list_directory
                    flags |= flag_backup_semantics
                elif for_read:
                    access |= generic_read
                    flags |= flag_sequential_scan
                handle = create_file(
                    _api_path(path),
                    access,
                    share_all,
                    None,
                    open_existing,
                    flags,
                    None,
                )
                if handle in (None, invalid_handle):
                    _raise_last_error("could not safely open trace path")
                try:
                    info = _win_info(handle)
                    kind = _win_kind(info)
                    if kind != expected_kind or not _stat_matches_handle(
                        before,
                        info,
                        kind,
                    ):
                        raise ValueError("trace node changed during safe open")
                    final_path = _normalized_final_path(handle)
                    return handle, info, final_path
                except BaseException:
                    close_handle(handle)
                    raise

            # A frame is (lexical path, stable handle, identity, final path,
            # kind).  Every active frame remains open until its subtree has
            # been fully captured and revalidated.
            frames: List[Tuple[Path, int, Tuple[Any, ...], str, str]] = []

            def _verify_frame(
                frame: Tuple[Path, int, Tuple[Any, ...], str, str]
            ) -> None:
                path, handle, expected, final_path, kind = frame
                current = _win_info(handle)
                if (
                    _win_kind(current) != kind
                    or not _same_win_object(current, expected, kind)
                ):
                    raise ValueError("trace handle identity drifted")
                if _normalized_final_path(handle) != final_path:
                    raise ValueError("trace handle path drifted")
                rebound_handle, rebound, rebound_path = _open_path(
                    path,
                    expected_kind=kind,
                )
                try:
                    if (
                        not _same_win_object(rebound, expected, kind)
                        or rebound_path != final_path
                    ):
                        raise ValueError("trace parent binding drifted")
                finally:
                    close_handle(rebound_handle)

            def _verify_chain() -> None:
                for frame in frames:
                    _verify_frame(frame)

            def _directory_names(path: Path) -> Tuple[str, ...]:
                _verify_chain()
                with os.scandir(path) as entries:
                    names = tuple(sorted(entry.name for entry in entries))
                _verify_chain()
                if any(
                    not name
                    or name in {".", ".."}
                    or "\\" in name
                    or "/" in name
                    or "\x00" in name
                    for name in names
                ):
                    raise ValueError("trace directory contains an unsafe name")
                return names

            def _read_windows_file(
                frame: Tuple[Path, int, Tuple[Any, ...], str, str]
            ) -> Tuple[bytes, str, int]:
                expected = frame[2]
                if expected[4] > max_trace_bytes - total:
                    raise ValueError("trace exceeds its configured byte limit")
                _verify_chain()
                captured = bytearray()
                digest = hashlib.sha256()
                read_total = 0
                while True:
                    amount = min(
                        TRACE_READ_CHUNK_BYTES,
                        max(1, max_trace_bytes - total - read_total + 1),
                    )
                    buffer = ctypes.create_string_buffer(amount)
                    received = wintypes.DWORD()
                    if not read_file(
                        frame[1],
                        buffer,
                        amount,
                        ctypes.byref(received),
                        None,
                    ):
                        _raise_last_error("could not read trace file")
                    if received.value == 0:
                        break
                    chunk = buffer.raw[: received.value]
                    read_total += len(chunk)
                    if total + read_total > max_trace_bytes:
                        raise ValueError("trace exceeds its configured byte limit")
                    digest.update(chunk)
                    captured.extend(chunk)
                _verify_chain()
                if read_total != expected[4]:
                    raise ValueError("trace file size drifted while being captured")
                return bytes(captured), digest.hexdigest(), read_total

            def _walk(
                frame: Tuple[Path, int, Tuple[Any, ...], str, str],
                relative_parts: Tuple[str, ...],
            ) -> None:
                nonlocal nodes
                nodes += 1
                if nodes > MAX_TRACE_NODES:
                    raise ValueError("trace exceeds its node limit")
                relative = "/".join(relative_parts)
                if frame[4] == "file":
                    data, digest, size = _read_windows_file(frame)
                    _append_file(relative, data, digest, size)
                    return
                names = _directory_names(frame[0])
                for name in names:
                    child_path = frame[0] / name
                    before = os.lstat(child_path)
                    child_kind = _assert_safe_node(before)
                    child_handle, child_info, child_final = _open_path(
                        child_path,
                        expected_kind=child_kind,
                        for_read=child_kind == "file",
                    )
                    if not _within(child_final, frame[3]):
                        close_handle(child_handle)
                        raise ValueError("trace child escaped its parent")
                    child_frame = (
                        child_path,
                        child_handle,
                        child_info,
                        child_final,
                        child_kind,
                    )
                    frames.append(child_frame)
                    try:
                        _walk(child_frame, relative_parts + (name,))
                        _verify_frame(child_frame)
                    finally:
                        frames.pop()
                        close_handle(child_handle)
                    _verify_chain()
                if _directory_names(frame[0]) != names:
                    raise ValueError("trace directory contents drifted")

            try:
                workspace_path = Path(os.path.abspath(os.fspath(workspace)))
                if not workspace_path.is_absolute() or not workspace_path.anchor:
                    raise ValueError("workspace path is not a Windows absolute path")

                current_path = Path(workspace_path.anchor)
                anchor_handle, anchor_info, anchor_final = _open_path(
                    current_path,
                    expected_kind="directory",
                )
                frames.append(
                    (
                        current_path,
                        anchor_handle,
                        anchor_info,
                        anchor_final,
                        "directory",
                    )
                )
                for part in workspace_path.parts[1:]:
                    parent_final = frames[-1][3]
                    current_path = current_path / part
                    handle, info, final_path = _open_path(
                        current_path,
                        expected_kind="directory",
                    )
                    if not _within(final_path, parent_final):
                        close_handle(handle)
                        raise ValueError("workspace parent contains a reparse escape")
                    frames.append(
                        (current_path, handle, info, final_path, "directory")
                    )

                for index, part in enumerate(trace_parts):
                    parent_final = frames[-1][3]
                    current_path = current_path / part
                    before = os.lstat(current_path)
                    kind = _assert_safe_node(before)
                    if index != len(trace_parts) - 1 and kind != "directory":
                        raise ValueError("trace parent is not a directory")
                    handle, info, final_path = _open_path(
                        current_path,
                        expected_kind=kind,
                        for_read=index == len(trace_parts) - 1 and kind == "file",
                    )
                    if not _within(final_path, parent_final):
                        close_handle(handle)
                        raise ValueError("trace path escaped its parent")
                    frames.append((current_path, handle, info, final_path, kind))

                _walk(frames[-1], trace_parts)
                _verify_chain()
            finally:
                for _, handle, _, _, _ in reversed(frames):
                    close_handle(handle)

        if os.name == "posix":
            _capture_posix()
        elif os.name == "nt":
            _capture_windows()
        else:
            raise OSError("secure trace capture is unsupported on this platform")

        return {
            "schema_version": TRACE_CAPTURE_SCHEMA_VERSION,
            "captured": True,
            "trace_ref": trace_ref.to_dict(),
            "files": files,
            "total_bytes": total,
            "content_omitted": False,
        }
    except Exception as exc:
        return {
            "schema_version": TRACE_CAPTURE_SCHEMA_VERSION,
            "captured": False,
            "reason": _safe_diag_text(exc.__class__.__name__),
            "trace_ref": trace_ref.to_dict(),
            "files": [],
            "total_bytes": None,
        }


@contextmanager
def _workspace_scope(handle: Any) -> Iterator[Tuple[Path, Any]]:
    """Normalize the real TrialWorkspace and small test doubles."""

    if isinstance(handle, TrialWorkspace):
        with handle as entered:
            yield entered.path, entered
        return
    if hasattr(handle, "__enter__") and hasattr(handle, "__exit__"):
        entered = handle.__enter__()
        try:
            path = getattr(entered, "path", entered)
            yield Path(path), entered
        finally:
            handle.__exit__(None, None, None)
        return
    path = getattr(handle, "path", handle)
    yield Path(path), handle


class EvalRunner:
    """Execute canonical Agent Trials and commit exactly one terminal result."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        repository_preparer: Optional[RepositoryPreparer],
        adapter: AgentUnderTestAdapter,
        case_provider: Any = None,
        *,
        matcher_factory: Any = None,
        capability_policy: CapabilityPolicy = CapabilityPolicy.STRICT,
        workspace_factory: Optional[Callable[..., Any]] = None,
        adapter_factory: Optional[Callable[[], AgentUnderTestAdapter]] = None,
        max_workers: Optional[int] = None,
        retry_incomplete: bool = True,
    ) -> None:
        if not isinstance(artifact_store, ArtifactStore):
            raise TypeError("artifact_store must be an ArtifactStore")
        if not isinstance(adapter, AgentUnderTestAdapter):
            raise TypeError("adapter must implement AgentUnderTestAdapter")
        if not _adapter_supports_cancellation(adapter):
            raise TypeError("adapter.run must accept the cancel_event keyword")
        if repository_preparer is not None and not isinstance(
            repository_preparer, RepositoryPreparer
        ):
            # A small structural fake is useful in unit tests; real production
            # instances still get the stronger type above at method boundaries.
            if not hasattr(repository_preparer, "prepare") or not hasattr(
                repository_preparer, "trial_workspace"
            ):
                raise TypeError("repository_preparer has no preparation interface")
        if not isinstance(capability_policy, CapabilityPolicy):
            raise TypeError("capability_policy must be a CapabilityPolicy")
        if max_workers is not None and (type(max_workers) is not int or max_workers < 1):
            raise ValueError("max_workers must be a positive integer or None")
        if type(retry_incomplete) is not bool:
            raise TypeError("retry_incomplete must be a bool")
        self.artifact_store = artifact_store
        self.repository_preparer = repository_preparer
        self.adapter = adapter
        self.case_provider = case_provider
        self.matcher_factory = matcher_factory or BuiltinMaterialClaimMatcherFactory()
        self.capability_policy = capability_policy
        self.workspace_factory = workspace_factory
        self.adapter_factory = adapter_factory
        self.max_workers = max_workers
        self.retry_incomplete = retry_incomplete
        self._cancel_event = threading.Event()
        # ArtifactStore intentionally uses non-blocking filesystem locks to
        # detect competing writers.  The bounded worker pool is one trusted
        # writer, so serialize its terminal publications before entering the
        # run-wide execution-budget namespace.
        self._submission_commit_lock = threading.Lock()

    def cancel(self) -> None:
        """Request cooperative cancellation of not-yet-committed Trials."""

        self._cancel_event.set()

    def preflight(
        self,
        config: EvalRunConfig,
        case_snapshot: Optional[RunCaseSnapshot] = None,
        *,
        policy: Optional[CapabilityPolicy] = None,
    ) -> CapabilityPreflight:
        config, case_snapshot = self._verified_inputs(config, case_snapshot)
        selected_policy = policy or self.capability_policy
        if not isinstance(selected_policy, CapabilityPolicy):
            raise TypeError("policy must be a CapabilityPolicy")
        adapter_id, adapter_version = _adapter_identity(self.adapter)
        issues: List[CapabilityIssue] = []
        checked: List[Tuple[str, int]] = []
        task_issues: Dict[str, bool] = {}
        for case in config.suite.cases:
            eval_input = case_snapshot.eval_input(case.task_id)
            for trial_index in range(1, config.trial_count + 1):
                checked.append((case.task_id, trial_index))
                binding = AgentRunConfig.bind(config, eval_input, trial_index)
                try:
                    result = self.adapter.compatibility(eval_input, binding)
                    if not isinstance(result, AdapterCompatibility):
                        raise TypeError("Adapter compatibility did not return AdapterCompatibility")
                    unsupported = tuple(
                        sorted(item.value for item in result.unsupported)
                    )
                    if unsupported:
                        task_issues[case.task_id] = True
                        issues.append(
                            CapabilityIssue(
                                task_id=case.task_id,
                                trial_index=trial_index,
                                reason="unsupported_input_capability",
                                unsupported=unsupported,
                                detail="Adapter does not support one or more canonical input capabilities",
                            )
                        )
                except AgentAdapterIncompatibleError as exc:
                    task_issues[case.task_id] = True
                    issues.append(
                        CapabilityIssue(
                            task_id=case.task_id,
                            trial_index=trial_index,
                            reason=exc.reason.value,
                            unsupported=(),
                            detail="Adapter declared this Trial incompatible",
                        )
                    )
                except Exception as exc:
                    task_issues[case.task_id] = True
                    issues.append(
                        CapabilityIssue(
                            task_id=case.task_id,
                            trial_index=trial_index,
                            reason="compatibility_check_failed",
                            unsupported=(),
                            detail=_safe_diag_text(exc.__class__.__name__),
                        )
                    )
        compatible = tuple(
            case.task_id for case in config.suite.cases if case.task_id not in task_issues
        )
        incompatible = tuple(
            case.task_id for case in config.suite.cases if case.task_id in task_issues
        )
        return CapabilityPreflight(
            run_id=config.run_id,
            agent_id=config.agent.agent_id,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            policy=selected_policy,
            checked_trials=tuple(checked),
            compatible_task_ids=compatible,
            incompatible_task_ids=incompatible,
            issues=tuple(issues),
        )

    def create_run(
        self,
        config: EvalRunConfig,
        case_snapshot: RunCaseSnapshot,
        *,
        policy: Optional[CapabilityPolicy] = None,
    ) -> RunSetup:
        """Preflight and create an immutable Run plan."""

        config, case_snapshot = self._verified_inputs(config, case_snapshot)
        selected_policy = policy or self.capability_policy
        preflight = self.preflight(config, case_snapshot, policy=selected_policy)
        if preflight.incompatible_task_ids and selected_policy is CapabilityPolicy.STRICT:
            config_digest = canonical_sha256(config)
            snapshot_digest = case_snapshot.digest()
            try:
                self.artifact_store.write_preflight_candidate(
                    config.run_id,
                    run_config_digest=config_digest,
                    case_snapshot_digest=snapshot_digest,
                    preflight=preflight.to_dict(),
                )
            except ArtifactConflictError as exc:
                existing = self.artifact_store.load_preflight_candidate(
                    config.run_id
                )
                if (
                    existing.get("run_config_digest") != config_digest
                    or existing.get("case_snapshot_digest") != snapshot_digest
                    or existing.get("preflight") != preflight.to_dict()
                ):
                    raise ArtifactIntegrityError(
                        "strict preflight audit conflicts with existing bytes"
                    ) from exc
            raise RunIncompatibilityError(
                "Adapter capability preflight rejected the Run",
                preflight=preflight,
                config=config,
                snapshot=case_snapshot,
            )
        if preflight.incompatible_task_ids:
            if not preflight.compatible_task_ids:
                raise RunIncompatibilityError(
                    "Adapter capability filter would produce an empty Run",
                    preflight=preflight,
                    config=config,
                    snapshot=case_snapshot,
                )
            case_snapshot = case_snapshot.select(preflight.compatible_task_ids)
            filtered_suite = SuiteRunConfig.from_case_snapshot(case_snapshot)
            suffix = "-capfilter-" + preflight.digest[:12]
            run_key = config.run_instance_key
            if len(run_key) + len(suffix) > 512:
                run_key = run_key[: 512 - len(suffix)]
            config = EvalRunConfig.create(
                run_instance_key=run_key + suffix,
                agent=config.agent,
                clarification_matcher=config.clarification_matcher,
                evaluator=config.evaluator,
                suite=filtered_suite,
                trial_count=config.trial_count,
                    resource_budgets=config.resource_budgets,
                )
            source_preflight = preflight
            final_preflight = self.preflight(
                config,
                case_snapshot,
                policy=CapabilityPolicy.FILTER,
            )
            if final_preflight.incompatible_task_ids:
                raise RunIncompatibilityError(
                    "filtered Run failed final Adapter capability preflight",
                    preflight=final_preflight,
                    config=config,
                    snapshot=case_snapshot,
                )
            preflight = self._filtered_preflight(
                source_preflight,
                final_preflight,
            )
        manifest = self.artifact_store.create_run(
            config,
            case_snapshot,
            run_preflight=preflight.to_dict(),
        )
        return RunSetup(
            config=config,
            case_snapshot=case_snapshot,
            manifest=manifest,
            preflight=preflight,
        )

    def run(
        self,
        config_or_run_id: EvalRunConfig | str,
        case_snapshot: Optional[RunCaseSnapshot] = None,
        *,
        policy: Optional[CapabilityPolicy] = None,
        resume: Optional[bool] = None,
        cancel_event: Optional[threading.Event] = None,
        max_workers: Optional[int] = None,
    ) -> RunResult:
        """Run a new config or resume an existing immutable Run."""

        selected_policy = policy or self.capability_policy
        should_resume = self.retry_incomplete if resume is None else resume
        if type(should_resume) is not bool:
            raise TypeError("resume must be a bool")
        if isinstance(config_or_run_id, EvalRunConfig):
            if case_snapshot is None:
                raise TypeError(
                    "a new EvalRunConfig requires its verified case_snapshot"
                )
            self._validate_parallel_configuration(config_or_run_id, max_workers)
            setup = self.create_run(
                config_or_run_id,
                case_snapshot,
                policy=selected_policy,
            )
            return self._execute(
                setup.config,
                setup.case_snapshot,
                setup.manifest,
                setup.preflight,
                created=True,
                resume=should_resume,
                cancel_event=cancel_event,
                max_workers=max_workers,
            )
        if type(config_or_run_id) is not str:
            raise TypeError("run requires EvalRunConfig or run_id string")
        config = self.artifact_store.load_run_config(config_or_run_id)
        snapshot = self.artifact_store.load_case_snapshot(config_or_run_id)
        self._validate_parallel_configuration(config, max_workers)
        try:
            preflight = CapabilityPreflight.from_dict(
                self.artifact_store.load_run_preflight(config_or_run_id)
            )
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "Run capability preflight payload is invalid"
            ) from exc
        self._validate_persisted_preflight(config, snapshot, preflight)
        if policy is not None and policy is not preflight.policy:
            raise RunnerError(
                "resume policy does not match the immutable capability preflight"
            )
        current_identity = _adapter_identity(self.adapter)
        if current_identity != (preflight.adapter_id, preflight.adapter_version):
            raise RunIncompatibilityError(
                "current Adapter identity does not match the immutable Run",
                preflight=preflight,
                config=config,
                snapshot=snapshot,
            )
        current_preflight = self.preflight(
            config,
            snapshot,
            policy=preflight.policy,
        )
        if current_preflight.incompatible_task_ids:
            # An existing immutable Run cannot be silently replaced by a
            # filtered plan.  Call create_run with the original config to get
            # the documented new identity instead.
            raise RunIncompatibilityError(
                "existing Run is incompatible with the Adapter",
                preflight=current_preflight,
                config=config,
                snapshot=snapshot,
            )
        manifest = self.artifact_store.load_run_manifest(config_or_run_id)
        return self._execute(
            config,
            snapshot,
            manifest,
            preflight,
            created=False,
            resume=should_resume,
            cancel_event=cancel_event,
            max_workers=max_workers,
        )

    # Explicit aliases make the CLI layer and callers read naturally while
    # preserving one implementation of the lifecycle.
    run_agent = run
    execute = run

    def _verified_inputs(
        self,
        config: EvalRunConfig,
        case_snapshot: Optional[RunCaseSnapshot],
    ) -> Tuple[EvalRunConfig, RunCaseSnapshot]:
        if not isinstance(config, EvalRunConfig):
            raise TypeError("config must be an EvalRunConfig")
        snapshot = case_snapshot or self._snapshot_for_config(config)
        if not isinstance(snapshot, RunCaseSnapshot):
            raise TypeError("case_snapshot must be a RunCaseSnapshot")
        if config.suite != SuiteRunConfig.from_case_snapshot(snapshot):
            raise SchemaError("Run Config suite does not match Case Snapshot")
        return config, snapshot

    def _validate_parallel_configuration(
        self,
        config: EvalRunConfig,
        max_workers: Optional[int],
    ) -> None:
        if max_workers is not None and (
            type(max_workers) is not int or max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer")
        worker_count = (
            max_workers
            if max_workers is not None
            else config.resource_budgets.max_parallel_trials
        )
        worker_count = min(worker_count, config.resource_budgets.max_parallel_trials)
        if (
            worker_count > 1
            and self.adapter_factory is None
        ):
            raise RunnerError(
                "parallel Trials require a per-Trial adapter_factory"
            )

    def _snapshot_for_config(self, config: EvalRunConfig) -> RunCaseSnapshot:
        return self.artifact_store.load_case_snapshot(config.run_id)

    @staticmethod
    def _filtered_preflight(
        source: CapabilityPreflight,
        final: CapabilityPreflight,
    ) -> CapabilityPreflight:
        if final.incompatible_task_ids:
            raise ValueError("final filtered preflight must be compatible")
        if source.run_id == final.run_id:
            raise ValueError("filtered preflight must bind a new Run identity")
        return CapabilityPreflight(
            run_id=final.run_id,
            agent_id=final.agent_id,
            adapter_id=final.adapter_id,
            adapter_version=final.adapter_version,
            policy=CapabilityPolicy.FILTER,
            checked_trials=final.checked_trials,
            compatible_task_ids=final.compatible_task_ids,
            incompatible_task_ids=final.incompatible_task_ids,
            # Preserve why Cases were removed while taking final coverage and
            # identity only from the second, fail-closed preflight.
            issues=source.issues + final.issues,
            filtered_from_run_id=source.run_id,
        )

    @staticmethod
    def _validate_persisted_preflight(
        config: EvalRunConfig,
        snapshot: RunCaseSnapshot,
        preflight: CapabilityPreflight,
    ) -> None:
        expected_trials = tuple(
            (case.task_id, trial_index)
            for case in config.suite.cases
            for trial_index in range(1, config.trial_count + 1)
        )
        expected_tasks = tuple(case.task_id for case in config.suite.cases)
        if (
            preflight.run_id != config.run_id
            or preflight.agent_id != config.agent.agent_id
            or preflight.checked_trials != expected_trials
            or preflight.compatible_task_ids != expected_tasks
            or preflight.incompatible_task_ids
            or config.suite != SuiteRunConfig.from_case_snapshot(snapshot)
        ):
            raise ArtifactIntegrityError(
                "Run capability preflight does not match the immutable plan"
            )

    def _execute(
        self,
        config: EvalRunConfig,
        snapshot: RunCaseSnapshot,
        manifest: RunManifest,
        preflight: CapabilityPreflight,
        *,
        created: bool,
        resume: bool,
        cancel_event: Optional[threading.Event],
        max_workers: Optional[int],
    ) -> RunResult:
        if cancel_event is not None and not hasattr(cancel_event, "is_set"):
            raise TypeError("cancel_event must provide is_set()")
        effective_cancel = _CombinedCancelEvent(self._cancel_event, cancel_event)
        prepared: Dict[str, PreparedRepository] = {}
        preparation_errors: Dict[str, BaseException] = {}
        descriptors: Dict[str, Any] = {}
        for entry in snapshot.cases:
            key = entry.input.repository.digest()
            descriptors[key] = entry.input.repository
        if self.repository_preparer is not None:
            for key, descriptor in descriptors.items():
                try:
                    value = self.repository_preparer.prepare(descriptor)
                    if not isinstance(value, PreparedRepository):
                        raise TypeError("repository preparer returned an invalid handle")
                    prepared[key] = value
                except BaseException as exc:
                    preparation_errors[key] = exc
        elif self.workspace_factory is None:
            error = RunnerError("no RepositoryPreparer or workspace_factory is configured")
            preparation_errors.update({key: error for key in descriptors})

        plans = tuple(manifest.trials)
        worker_count = max_workers if max_workers is not None else self.max_workers
        if worker_count is None:
            worker_count = config.resource_budgets.max_parallel_trials
        if type(worker_count) is not int or worker_count < 1:
            raise ValueError("max_workers must be a positive integer")
        worker_count = min(worker_count, config.resource_budgets.max_parallel_trials)
        if (
            worker_count > 1
            and self.adapter_factory is None
        ):
            raise RunnerError(
                "parallel Trials require a per-Trial adapter_factory"
            )

        def invoke(plan: Any) -> TrialResult:
            return self._execute_trial(
                config,
                snapshot,
                plan,
                preflight,
                prepared,
                preparation_errors,
                resume=resume,
                cancel_event=effective_cancel,
            )

        results: List[TrialResult] = []
        if worker_count == 1 or len(plans) <= 1:
            for plan in plans:
                try:
                    results.append(invoke(plan))
                except BaseException as exc:
                    results.append(
                        self._catastrophic_result(
                            config,
                            snapshot,
                            plan,
                            exc,
                            resume=resume,
                        )
                    )
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="eval-trial",
            ) as executor:
                futures: Dict[Future[TrialResult], Any] = {
                    executor.submit(invoke, plan): plan for plan in plans
                }
                for future in as_completed(futures):
                    plan = futures[future]
                    try:
                        results.append(future.result())
                    except BaseException as exc:
                        # A worker should normally convert every exception to a
                        # Submission.  Preserve a deterministic diagnostic if
                        # a bug escapes the boundary instead of dropping Case.
                        results.append(
                            self._catastrophic_result(
                                config, snapshot, plan, exc, resume=resume
                            )
                        )
        ordered = tuple(sorted(results, key=lambda item: (item.task_id, item.trial_index)))
        status = self.artifact_store.load_run_state(config.run_id).status
        return RunResult(
            run_id=config.run_id,
            config=config,
            preflight=preflight,
            trials=ordered,
            status=status,
            created=created,
        )

    def _new_adapter(
        self,
        expected_identity: Tuple[str, str],
    ) -> AgentUnderTestAdapter:
        try:
            value = (
                self.adapter
                if self.adapter_factory is None
                else self.adapter_factory()
            )
        except BaseException as exc:
            raise _AdapterFactoryError("Adapter factory failed") from exc
        if not isinstance(value, AgentUnderTestAdapter):
            raise _AdapterFactoryError("adapter_factory returned an invalid Adapter")
        if not _adapter_supports_cancellation(value):
            raise _AdapterFactoryError(
                "adapter_factory returned a non-cancellable Adapter"
            )
        if _adapter_identity(value) != expected_identity:
            raise _AdapterIdentityMismatch(ADAPTER_IDENTITY_MISMATCH)
        return value

    def _make_workspace(
        self,
        *,
        prepared: Optional[PreparedRepository],
        trial_manifest: TrialManifest,
        suite_case: Any,
        eval_input: EvalInput,
        attempt: int,
    ) -> Any:
        if self.workspace_factory is not None:
            return self.workspace_factory(
                prepared_repository=prepared,
                trial_manifest=trial_manifest,
                suite_case=suite_case,
                eval_input=eval_input,
                attempt=attempt,
            )
        if self.repository_preparer is None or prepared is None:
            raise RunnerError("isolated workspace cannot be prepared")
        return self.repository_preparer.trial_workspace(
            prepared,
            trial_manifest=trial_manifest,
            suite_case=suite_case,
            eval_input=eval_input,
            attempt=attempt,
        )

    def _execute_trial(
        self,
        config: EvalRunConfig,
        snapshot: RunCaseSnapshot,
        plan: Any,
        preflight: CapabilityPreflight,
        prepared: Mapping[str, PreparedRepository],
        preparation_errors: Mapping[str, BaseException],
        *,
        resume: bool,
        cancel_event: Any,
    ) -> TrialResult:
        trial_manifest = self.artifact_store.load_trial_manifest(
            config.run_id, plan.task_id, plan.trial_id
        )
        eval_input = snapshot.eval_input(plan.task_id)
        suite_case = config.suite.case(plan.task_id)
        state = self.artifact_store.load_trial_state(
            config.run_id, plan.task_id, plan.trial_id
        )
        if state.status in {
            TrialStatus.COMPLETED,
            TrialStatus.FAILED,
            TrialStatus.BLOCKED,
            TrialStatus.INVALID_OUTPUT,
        }:
            return self._existing_result(config, trial_manifest, state)
        if state.status in {TrialStatus.RUNNING, TrialStatus.INCOMPLETE}:
            # A process that disappeared between turns is never rerun from a
            # stale lease.  Recovery first marks an active attempt incomplete
            # and adopts only canonical orphan artifacts before a new attempt
            # may start.
            self.artifact_store.recover_trial(
                config.run_id, plan.task_id, plan.trial_id
            )
            state = self.artifact_store.load_trial_state(
                config.run_id, plan.task_id, plan.trial_id
            )
            if state.status in {
                TrialStatus.COMPLETED,
                TrialStatus.FAILED,
                TrialStatus.BLOCKED,
                TrialStatus.INVALID_OUTPUT,
            }:
                return self._existing_result(config, trial_manifest, state)
        if state.status is TrialStatus.INCOMPLETE and not resume:
            return self._nonterminal_result(config, trial_manifest, state, "resume disabled")

        if cancel_event.is_set():
            return self._commit_failure_without_workspace(
                config,
                snapshot,
                trial_manifest,
                suite_case,
                FailureCode.PROCESS_KILLED,
                attempt_hint=state.active_attempt,
                diagnostic="cancelled before Trial invocation",
                preflight=preflight,
            )

        try:
            running = self.artifact_store.start_trial(
                config.run_id, plan.task_id, plan.trial_id
            )
        except ArtifactConflictError:
            latest = self.artifact_store.load_trial_state(
                config.run_id, plan.task_id, plan.trial_id
            )
            if latest.status in {
                TrialStatus.COMPLETED,
                TrialStatus.FAILED,
                TrialStatus.BLOCKED,
                TrialStatus.INVALID_OUTPUT,
            }:
                return self._existing_result(config, trial_manifest, latest)
            raise
        attempt = running.active_attempt
        if attempt is None:
            raise RunnerError("started Trial has no active attempt")
        binding = AgentRunConfig.bind(config, eval_input, trial_manifest.trial_index)
        if binding.trial_id != trial_manifest.trial_id:
            raise RunnerError("AgentRunConfig trial binding drifted")

        repository_key = eval_input.repository.digest()
        preparation_error = preparation_errors.get(repository_key)
        workspace_handle: Any = None
        workspace_path: Optional[Path] = None
        workspace_binding_id: Optional[str] = None
        started = time.monotonic()
        controller = _LazyClarificationSession(
            task_id=plan.task_id,
            provider=self.case_provider,
            binding=binding,
            matcher_factory=self.matcher_factory,
        )
        submission: Optional[EvalSubmission] = None
        incompatibility: Optional[str] = None
        diagnostic = ""
        terminal_status: Optional[TrialStatus] = None

        try:
            if preparation_error is not None:
                raise RunnerError("repository preparation failed")
            workspace_handle = self._make_workspace(
                prepared=prepared.get(repository_key),
                trial_manifest=trial_manifest,
                suite_case=suite_case,
                eval_input=eval_input,
                attempt=attempt,
            )
            with _workspace_scope(workspace_handle) as (workspace, entered):
                workspace_binding_id = getattr(
                    getattr(entered, "manifest", None), "workspace_binding_id", None
                )
                if type(workspace_binding_id) is not str or not workspace_binding_id:
                    raise RunnerError(
                        "workspace did not expose a canonical target materialization ID"
                    )
                if not isinstance(workspace, Path):
                    raise RunnerError("workspace handle did not provide a Path")
                workspace = workspace.resolve(strict=True)
                workspace_path = workspace
                current_state = self.artifact_store.load_trial_state(
                    config.run_id, plan.task_id, plan.trial_id
                )
                if StageName.PREPARE not in current_state.completed_stages:
                    self.artifact_store.write_prepare_stage(
                        config.run_id,
                        plan.task_id,
                        plan.trial_id,
                        eval_input,
                        attempt=attempt,
                    )
                if cancel_event.is_set():
                    submission = self._failure_submission(
                        eval_input,
                        binding,
                        FailureCode.PROCESS_KILLED,
                        target_materialization_id=workspace_binding_id,
                        elapsed=time.monotonic() - started,
                        retryable=False,
                    )
                else:
                    try:
                        adapter = self._new_adapter(
                            (preflight.adapter_id, preflight.adapter_version)
                        )
                        candidate = _invoke_adapter(
                            adapter,
                            eval_input,
                            workspace,
                            binding,
                            controller.channel,
                            cancel_event,
                            target_materialization_id=workspace_binding_id,
                        )
                        if cancel_event.is_set():
                            submission = self._failure_submission(
                                eval_input,
                                binding,
                                FailureCode.PROCESS_KILLED,
                                target_materialization_id=workspace_binding_id,
                                elapsed=time.monotonic() - started,
                                retryable=False,
                            )
                        elif not isinstance(candidate, EvalSubmission):
                            raise AgentAdapterError(
                                FailureCode.SCHEMA_MISMATCH,
                                "Adapter did not return EvalSubmission",
                                retryable=False,
                            )
                        else:
                            if len(canonical_json_bytes(candidate)) > binding.max_output_bytes:
                                raise AgentAdapterError(
                                    FailureCode.OUTPUT_OVERFLOW,
                                    "Adapter Submission exceeds the configured output limit",
                                    retryable=False,
                                )
                            candidate = validate_submission_binding(
                                candidate,
                                eval_input=eval_input,
                                config=binding,
                                target_materialization_id=workspace_binding_id,
                                clarification_transcript=controller.transcript,
                            )
                            candidate = validate_submission_trace(
                                candidate,
                                workspace=workspace,
                                max_trace_bytes=binding.max_trace_bytes,
                            )
                            submission = candidate
                    except _AdapterIdentityMismatch:
                        incompatibility = ADAPTER_IDENTITY_MISMATCH
                        self.artifact_store.mark_trial_incomplete(
                            config.run_id,
                            plan.task_id,
                            plan.trial_id,
                            attempt=attempt,
                        )
                        terminal_status = TrialStatus.INCOMPLETE
                        diagnostic = "Adapter identity drifted after preflight"
                    except AgentAdapterIncompatibleError as exc:
                        incompatibility = exc.reason.value
                        self.artifact_store.mark_trial_incomplete(
                            config.run_id,
                            plan.task_id,
                            plan.trial_id,
                            attempt=attempt,
                        )
                        terminal_status = TrialStatus.INCOMPLETE
                        diagnostic = "dynamic capability incompatibility"
                    except RunnerError:
                        # Script/matcher/factory failures belong to the
                        # Harness boundary and are handled by the outer
                        # incomplete path, never scored as Agent output.
                        raise
                    except AgentAdapterError as exc:
                        submission = self._failure_submission(
                            eval_input,
                            binding,
                            exc.code,
                            target_materialization_id=workspace_binding_id,
                            elapsed=time.monotonic() - started,
                            retryable=exc.retryable,
                        )
                    except ClarificationProtocolError:
                        submission = self._failure_submission(
                            eval_input,
                            binding,
                            FailureCode.ADAPTER_ERROR,
                            target_materialization_id=workspace_binding_id,
                            elapsed=time.monotonic() - started,
                            retryable=False,
                        )
                    except BaseException as exc:
                        submission = self._failure_submission(
                            eval_input,
                            binding,
                            _exception_failure_code(exc),
                            target_materialization_id=workspace_binding_id,
                            elapsed=time.monotonic() - started,
                            retryable=_exception_failure_code(exc)
                            in {FailureCode.TIMEOUT, FailureCode.PROCESS_KILLED},
                        )
                    if submission is not None:
                        terminal_status = _terminal_status_for_submission(submission)
                        diagnostic = (
                            "completed Submission"
                            if submission.status is SubmissionStatus.COMPLETED
                            else _failure_message(submission.failure.code)
                            if submission.failure is not None
                            else "terminal Submission"
                        )
                if (
                    entered is not None
                    and hasattr(entered, "record_terminal_status")
                    and terminal_status is not None
                ):
                    entered.record_terminal_status(terminal_status)
                elif (
                    entered is not None
                    and hasattr(entered, "record_terminal_status")
                    and incompatibility is not None
                ):
                    entered.record_terminal_status(TrialStatus.INCOMPLETE)

                if submission is not None:
                    self._commit_submission(
                        config,
                        trial_manifest,
                        submission,
                        attempt=attempt,
                        controller=controller,
                        preflight=preflight,
                        workspace_handle=entered or workspace_handle,
                        workspace_path=workspace_path,
                        adapter=(adapter if "adapter" in locals() else self.adapter),
                        elapsed=time.monotonic() - started,
                    )
        except AgentAdapterIncompatibleError as exc:
            incompatibility = exc.reason.value
            self.artifact_store.mark_trial_incomplete(
                config.run_id,
                plan.task_id,
                plan.trial_id,
                attempt=attempt,
            )
            terminal_status = TrialStatus.INCOMPLETE
            diagnostic = "dynamic capability incompatibility"
        except BaseException as exc:
            # Exceptions that escape the inner Adapter boundary come from
            # repository/workspace preparation, artifact publication, or the
            # Runner itself.  They are Harness failures, not Agent failures,
            # and therefore must not create a scored ``adapter_error``.
            diagnostic = "harness failure: " + _safe_diag_text(
                exc.__class__.__name__
            )
            latest = self.artifact_store.load_trial_state(
                config.run_id,
                plan.task_id,
                plan.trial_id,
            )
            if latest.status is TrialStatus.RUNNING:
                self.artifact_store.mark_trial_incomplete(
                    config.run_id,
                    plan.task_id,
                    plan.trial_id,
                    attempt=attempt,
                )
                terminal_status = TrialStatus.INCOMPLETE
            elif latest.status is TrialStatus.INCOMPLETE:
                terminal_status = TrialStatus.INCOMPLETE
            else:
                terminal_status = latest.status
            submission = None

        if incompatibility is not None:
            state = self.artifact_store.load_trial_state(
                config.run_id, plan.task_id, plan.trial_id
            )
            return TrialResult(
                run_id=config.run_id,
                task_id=plan.task_id,
                trial_id=plan.trial_id,
                trial_index=plan.trial_index,
                status=state.status,
                submission=None,
                attempt=attempt,
                skipped=False,
                workspace_binding_id=workspace_binding_id,
                incompatibility=incompatibility,
                diagnostic=diagnostic,
            )
        final_state = self.artifact_store.load_trial_state(
            config.run_id, plan.task_id, plan.trial_id
        )
        if final_state.terminal_receipt is not None:
            submission = self.artifact_store.load_existing_submission(
                config.run_id, plan.task_id, plan.trial_id
            )
        return TrialResult(
            run_id=config.run_id,
            task_id=plan.task_id,
            trial_id=plan.trial_id,
            trial_index=plan.trial_index,
            status=final_state.status,
            submission=submission,
            attempt=attempt,
            skipped=False,
            workspace_binding_id=workspace_binding_id,
            incompatibility=None,
            diagnostic=diagnostic,
        )

    def _commit_submission(
        self,
        config: EvalRunConfig,
        trial_manifest: TrialManifest,
        submission: EvalSubmission,
        *,
        attempt: int,
        controller: _LazyClarificationSession,
        preflight: CapabilityPreflight,
        workspace_handle: Any,
        workspace_path: Optional[Path],
        adapter: Any,
        elapsed: float,
    ) -> TrialState:
        diagnostic = _read_adapter_diagnostic(adapter)
        failure_code = None if submission.failure is None else submission.failure.code.value
        preflight_ref = {
            "schema_version": CAPABILITY_PREFLIGHT_SCHEMA_VERSION,
            "run_id": config.run_id,
            "receipt_path": "receipts/capability_preflight.json",
            "preflight_digest": preflight.digest,
            "coverage": preflight.coverage,
        }
        clarification_receipt = _clarification_artifact(
            AgentRunConfig.bind(
                config,
                self.artifact_store.load_case_snapshot(config.run_id).eval_input(
                    trial_manifest.task_id
                ),
                trial_manifest.trial_index,
            ),
            controller,
        )
        runner_artifacts: Dict[str, Any] = {
            "capability_preflight.json": preflight_ref,
            "clarification_match_receipts.json": clarification_receipt,
            "terminal_summary.json": {
                "schema_version": TERMINAL_SUMMARY_SCHEMA_VERSION,
                "run_id": config.run_id,
                "task_id": trial_manifest.task_id,
                "trial_id": trial_manifest.trial_id,
                "attempt": attempt,
                "status": submission.status.value,
                "failure_code": failure_code,
                "elapsed_seconds": max(0.0, float(elapsed)),
                "stdout": _stream_summary(diagnostic.stdout, diagnostic.stdout_bytes),
                "stderr": _stream_summary(diagnostic.stderr, diagnostic.stderr_bytes),
                "adapter_id": _adapter_identity(adapter)[0],
                "adapter_version": _adapter_identity(adapter)[1],
            },
        }
        workspace_value = _workspace_manifest_value(workspace_handle)
        if workspace_value is not None:
            runner_artifacts["workspace_manifest.json"] = workspace_value
        trace_value = _capture_trace_summary(
            submission,
            workspace_path,
            max_trace_bytes=config.resource_budgets.max_trace_bytes,
        )
        if (
            submission.trace_ref is not None
            and submission.trace_ref.type is TraceType.LOCAL_PATH
            and (
                trace_value is None
                or trace_value.get("captured") is not True
            )
        ):
            raise RunnerError("local trace could not be captured safely")
        if trace_value is not None:
            runner_artifacts["trace_capture.json"] = trace_value
        with self._submission_commit_lock:
            try:
                return self.artifact_store.finalize_submission(
                    config.run_id,
                    trial_manifest.task_id,
                    trial_manifest.trial_id,
                    submission,
                    attempt=attempt,
                    runner_artifacts=runner_artifacts,
                )
            except ExecutionArtifactBudgetError:
                # A byte/count budget may discard optional diagnostics.  The
                # clarification receipt and any local trace capture are
                # required audit artifacts and bypass the execution budget;
                # integrity/security/conflict errors still propagate.
                required = {
                    "clarification_match_receipts.json": clarification_receipt,
                    "terminal_summary.json": runner_artifacts[
                        "terminal_summary.json"
                    ],
                }
                if trace_value is not None:
                    required["trace_capture.json"] = trace_value
                return self.artifact_store.finalize_submission(
                    config.run_id,
                    trial_manifest.task_id,
                    trial_manifest.trial_id,
                    submission,
                    attempt=attempt,
                    runner_artifacts=required,
                )

    def _failure_submission(
        self,
        eval_input: EvalInput,
        binding: AgentRunConfig,
        code: FailureCode,
        *,
        target_materialization_id: str,
        elapsed: float,
        retryable: bool,
    ) -> EvalSubmission:
        # ``clarification_required`` is only legal when the Adapter returns a
        # real unresolved exchange.  An exception alone carries no question
        # and must not make the Runner fabricate one; use the canonical
        # blocked-without-transcript code instead.
        if code is FailureCode.CLARIFICATION_REQUIRED:
            code = FailureCode.AGENT_BLOCKED
        return failure_submission(
            eval_input=eval_input,
            config=binding,
            target_materialization_id=target_materialization_id,
            code=code,
            message=_failure_message(code),
            retryable=retryable,
            usage=empty_usage(elapsed_seconds=max(0.0, elapsed)),
        )

    def _existing_result(
        self,
        config: EvalRunConfig,
        trial_manifest: TrialManifest,
        state: TrialState,
    ) -> TrialResult:
        submission = self.artifact_store.load_existing_submission(
            config.run_id, trial_manifest.task_id, trial_manifest.trial_id
        )
        return TrialResult(
            run_id=config.run_id,
            task_id=trial_manifest.task_id,
            trial_id=trial_manifest.trial_id,
            trial_index=trial_manifest.trial_index,
            status=state.status,
            submission=submission,
            attempt=state.active_attempt,
            skipped=True,
            workspace_binding_id=None,
            incompatibility=None,
            diagnostic="terminal Submission already exists",
        )

    @staticmethod
    def _nonterminal_result(
        config: EvalRunConfig,
        trial_manifest: TrialManifest,
        state: TrialState,
        diagnostic: str,
    ) -> TrialResult:
        return TrialResult(
            run_id=config.run_id,
            task_id=trial_manifest.task_id,
            trial_id=trial_manifest.trial_id,
            trial_index=trial_manifest.trial_index,
            status=state.status,
            submission=None,
            attempt=state.active_attempt,
            skipped=True,
            workspace_binding_id=None,
            incompatibility=None,
            diagnostic=diagnostic,
        )

    def _commit_failure_without_workspace(
        self,
        config: EvalRunConfig,
        snapshot: RunCaseSnapshot,
        trial_manifest: TrialManifest,
        suite_case: Any,
        code: FailureCode,
        *,
        attempt_hint: Optional[int],
        diagnostic: str,
        preflight: CapabilityPreflight,
    ) -> TrialResult:
        # A strict v2 Submission cannot be fabricated before a canonical
        # per-attempt target materialization exists.  Keep this attempt
        # recoverable; a resumed attempt will materialize first and can then
        # bind any terminal Submission to that real identity.
        del snapshot, suite_case, code, preflight
        state = self.artifact_store.load_trial_state(
            config.run_id, trial_manifest.task_id, trial_manifest.trial_id
        )
        if state.status is TrialStatus.INCOMPLETE and state.active_attempt is not None:
            attempt = state.active_attempt
            running = self.artifact_store.start_trial(
                config.run_id, trial_manifest.task_id, trial_manifest.trial_id
            )
            attempt = running.active_attempt or attempt
        elif state.status is TrialStatus.PENDING:
            running = self.artifact_store.start_trial(
                config.run_id, trial_manifest.task_id, trial_manifest.trial_id
            )
            attempt = running.active_attempt
        else:
            attempt = attempt_hint or state.active_attempt
        if attempt is None:
            raise RunnerError("cannot commit cancellation without a Trial attempt")
        final = self.artifact_store.load_trial_state(
            config.run_id, trial_manifest.task_id, trial_manifest.trial_id
        )
        if final.status is TrialStatus.RUNNING:
            final = self.artifact_store.mark_trial_incomplete(
                config.run_id,
                trial_manifest.task_id,
                trial_manifest.trial_id,
                attempt=attempt,
            )
        if final.status is not TrialStatus.INCOMPLETE:
            raise RunnerError(
                "pre-materialization cancellation did not remain incomplete"
            )
        return TrialResult(
            run_id=config.run_id,
            task_id=trial_manifest.task_id,
            trial_id=trial_manifest.trial_id,
            trial_index=trial_manifest.trial_index,
            status=final.status,
            submission=None,
            attempt=attempt,
            skipped=False,
            workspace_binding_id=None,
            incompatibility=None,
            diagnostic=diagnostic,
        )

    def _catastrophic_result(
        self,
        config: EvalRunConfig,
        snapshot: RunCaseSnapshot,
        plan: Any,
        exc: BaseException,
        *,
        resume: bool,
    ) -> TrialResult:
        del snapshot, resume
        state = self.artifact_store.load_trial_state(
            config.run_id, plan.task_id, plan.trial_id
        )
        if state.status in {
            TrialStatus.COMPLETED,
            TrialStatus.FAILED,
            TrialStatus.BLOCKED,
            TrialStatus.INVALID_OUTPUT,
        }:
            return self._existing_result(config, plan, state)
        trial_manifest = self.artifact_store.load_trial_manifest(
            config.run_id, plan.task_id, plan.trial_id
        )
        if state.status is TrialStatus.RUNNING and state.active_attempt is not None:
            state = self.artifact_store.mark_trial_incomplete(
                config.run_id,
                plan.task_id,
                plan.trial_id,
                attempt=state.active_attempt,
            )
        return self._nonterminal_result(
            config,
            trial_manifest,
            state,
            "Runner boundary failure: " + _safe_diag_text(exc.__class__.__name__),
        )


# Conventional names used by CLI integrations and older callers.
AgentRunner = EvalRunner
TrialRunner = EvalRunner


__all__ = [
    "RUNNER_SCHEMA_VERSION",
    "CAPABILITY_PREFLIGHT_SCHEMA_VERSION",
    "TERMINAL_SUMMARY_SCHEMA_VERSION",
    "CLARIFICATION_RECEIPT_SCHEMA_VERSION",
    "TRACE_CAPTURE_SCHEMA_VERSION",
    "ADAPTER_IDENTITY_MISMATCH",
    "RunnerError",
    "RunIncompatibilityError",
    "CapabilityPolicy",
    "PreflightMode",
    "CapabilityIssue",
    "CapabilityPreflight",
    "RunSetup",
    "TrialResult",
    "RunResult",
    "ClarificationScriptProvider",
    "AdapterDiagnostic",
    "EvalRunner",
    "AgentRunner",
    "TrialRunner",
]
