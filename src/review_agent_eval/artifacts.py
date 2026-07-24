"""Create-only, resumable artifacts for canonical code-review evaluations.

``run_manifest.json`` and every ``trial_manifest.json`` are immutable plans.
Execution state is derived only from create-only receipts.  In particular, a
Submission is committed by one terminal agent receipt written *after* all
artifacts it binds; nonterminal/incomplete Trials therefore have no committed
Submission.  Evaluator artifacts live under versioned ``evaluations``
namespaces and never mutate the agent Submission.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

from .cases import (
    MAX_RUN_CASE_SNAPSHOT_BYTES,
    PublicSuitePreparationBindingV2,
    RunCaseSnapshot,
    WireContractV2,
)
from .config import (
    MAX_EVAL_RUN_CONFIG_BYTES,
    EvalRunConfig,
    EvaluatorExecutionConfig,
    SuiteRunConfig,
    derive_case_path_id,
    derive_evaluation_id,
    derive_trial_id,
    derive_trial_seed,
    validate_case_path_id,
    validate_evaluation_id,
    validate_evaluation_id_shape,
    validate_path_segment,
    validate_run_id,
    validate_safe_json,
    validate_safe_text,
    validate_trial_id,
    validate_trial_id_shape,
    _EVALUATOR_CONTEXT_POLICIES,
)
from .models import (
    EVAL_SUBMISSION_SCHEMA_VERSION,
    MAX_COUNTER,
    MAX_EVAL_INPUT_BYTES,
    MAX_EVAL_SUBMISSION_BYTES,
    MAX_JSON_DEPTH,
    MAX_JSON_INTEGER_DIGITS,
    EvalInput,
    EvalSubmission,
    FailureCode,
    ReviewTargetKind,
    SchemaError,
    SubmissionFailure,
    SubmissionStatus,
    SubmissionUsage,
    TraceType,
    TrialStatus,
    UnsupportedProtocolVersionError,
    _JsonModel,
    _array,
    _check_model_size,
    _digest,
    _enum_value,
    _exact_fields,
    _identifier,
    _integer,
    _object,
    _strict_json_loads,
    canonical_json_bytes,
    stable_id,
    submission_status_for_failure,
)


EVAL_RUN_MANIFEST_SCHEMA_VERSION = "eval_run_manifest_v2"
EVAL_TRIAL_MANIFEST_SCHEMA_VERSION = "eval_trial_manifest_v2"
EVAL_STAGE_RECEIPT_SCHEMA_VERSION = "eval_stage_receipt_v2"
EVAL_RUN_PREFLIGHT_SCHEMA_VERSION = "eval_capability_preflight_v2"
EVAL_PREFLIGHT_CANDIDATE_SCHEMA_VERSION = "eval_preflight_candidate_v2"
EVAL_TRIAL_MATERIALIZATION_SCHEMA_VERSION = "eval_trial_materialization_v2"
PRE_MATERIALIZATION_FAILURE_BINDING_VERSION = (
    "eval_pre_materialization_failure_binding_v2"
)
EVAL_RUN_EVALUATION_NAMESPACE_SCHEMA_VERSION = (
    "eval_run_evaluation_namespace_v2"
)

MAX_RUN_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_TRIAL_MANIFEST_BYTES = 1024 * 1024
MAX_STAGE_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_TRIAL_MATERIALIZATION_BYTES = 64 * 1024 * 1024
MAX_RUN_PREFLIGHT_BYTES = MAX_RUN_CASE_SNAPSHOT_BYTES
MAX_PREFLIGHT_CANDIDATE_BYTES = MAX_RUN_CASE_SNAPSHOT_BYTES
MAX_MANIFEST_TRIALS = 100_000
MAX_RECEIPT_ARTIFACTS = 128
MAX_ARTIFACT_PATH_CHARS = 2_048
MAX_TRIAL_ATTEMPTS = 10_000
MAX_AGENT_VISIBLE_FILES = 65_536
DEFAULT_MAX_FILE_BYTES = MAX_RUN_CASE_SNAPSHOT_BYTES
DEFAULT_MAX_TOTAL_READ_BYTES = 512 * 1024 * 1024
# Runner-owned execution metadata is deliberately a small, named surface.
# Callers cannot use the terminal path as an untyped arbitrary artifact sink.
MAX_RUNNER_ARTIFACTS = 16
MAX_RUNNER_ARTIFACT_NAME_CHARS = 128
_RUNNER_ARTIFACT_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\.json$")
_KNOWN_RUNNER_ARTIFACT_NAMES = frozenset(
    {
        "capability_preflight.json",
        "clarification_match_receipts.json",
        "terminal_summary.json",
        "workspace_manifest.json",
        "command_attestations.json",
        "trace_capture.json",
    }
)
_CONTROL_RUNNER_ARTIFACT_NAMES = frozenset(
    {
        "clarification_match_receipts.json",
        "terminal_summary.json",
    }
)
_MANDATORY_EXECUTION_RUNNER_ARTIFACT_NAMES = frozenset(
    {"trace_capture.json"}
)
_REQUIRED_RUNNER_ARTIFACT_NAMES = (
    _CONTROL_RUNNER_ARTIFACT_NAMES
    | _MANDATORY_EXECUTION_RUNNER_ARTIFACT_NAMES
)
_EVALUATION_JSON_ARTIFACT_NAMES = (
    "evaluator_execution_config.json",
    "intent_matches.json",
    "review_matches.json",
    "judge_input.json",
    "judge_output.json",
    "score.json",
)
_EVALUATOR_CONTEXT_POLICY_BY_ARTIFACT = {
    "review_matches.json": "review_matches",
    "judge_input.json": "judge_input",
    "judge_output.json": "judge_output",
}
_EVALUATOR_CONTEXT_POLICY_BY_BUNDLE_FIELD = {
    "_review_matches_json": "review_matches",
    "_judge_input_json": "judge_input",
    "_judge_output_json": "judge_output",
}


def _evaluator_context_policy_for_payload(
    value: Any,
    candidate_policy: Optional[str],
) -> Optional[str]:
    if candidate_policy is None:
        return None
    policy = _EVALUATOR_CONTEXT_POLICIES.get(candidate_policy)
    if policy is None:
        return candidate_policy
    payload = value.to_dict() if isinstance(value, _JsonModel) else value
    if (
        isinstance(payload, Mapping)
        and payload.get("schema_version") == policy[0]
    ):
        return candidate_policy
    return None


_EVALUATION_OPTIONAL_ARTIFACT_NAMES = ("report.md",)
_EVALUATION_NAMESPACE_FILENAMES = frozenset(
    (
        *_EVALUATION_JSON_ARTIFACT_NAMES,
        *_EVALUATION_OPTIONAL_ARTIFACT_NAMES,
        "receipt.json",
        ".locks",
    )
)
_RUN_EVALUATION_FILENAMES = frozenset({"summary.json", "report.md"})
DIRECTORY_FSYNC_SUPPORTED = os.name != "nt"


_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_ATTEMPT_RE = re.compile(r"^attempt-[0-9]{4,5}$")
_MATERIALIZATION_ID_RE = re.compile(r"^materialization-[0-9a-f]{64}$")
_TEMP_FILE_RE = re.compile(r"^\..+\.[0-9a-f]{32}\.tmp$")
_INTERNAL_DIRECTORIES = frozenset(
    {
        ".locks",
        "cases",
        "trials",
        "receipts",
        "evaluations",
        "preflights",
        "materializations",
    }
)


def _bounded_tuple(
    values: Iterable[Any], context: str, maximum: int
) -> Tuple[Any, ...]:
    try:
        items = tuple(islice(iter(values), maximum + 1))
    except TypeError as exc:
        raise SchemaError("%s must be iterable" % context) from exc
    if len(items) > maximum:
        raise SchemaError("%s exceeds its item limit" % context)
    return items


_TERMINAL_STATUSES = frozenset(
    {
        TrialStatus.COMPLETED,
        TrialStatus.FAILED,
        TrialStatus.BLOCKED,
        TrialStatus.INVALID_OUTPUT,
    }
)
class ArtifactError(RuntimeError):
    """Base class for artifact persistence failures."""


class ArtifactConflictError(ArtifactError):
    """A create-only artifact or writer claim already exists."""


class ArtifactIntegrityError(ArtifactError):
    """Stored bytes fail canonical, size, hash, or identity validation."""


class ExecutionArtifactBudgetError(ArtifactIntegrityError):
    """Optional execution metadata exceeded its configured byte/count budget."""


class RequiredExecutionArtifactBudgetError(ArtifactIntegrityError):
    """A mandatory trace/attestation cannot fit the execution budget."""


class ArtifactSecurityError(ArtifactIntegrityError):
    """A path contains a symlink, reparse point, or special node."""


class ArtifactStateError(ArtifactError):
    """The requested receipt would violate the Trial state machine."""


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    INCOMPLETE = "incomplete"
    COMPLETED = "completed"


class StageName(str, Enum):
    START = "start"
    PREPARE = "prepare"
    INCOMPLETE = "incomplete"
    AGENT = "agent"
    EVALUATOR = "evaluator"


def _require_enum(enum_type: Any, value: Any, context: str) -> Any:
    if not isinstance(value, enum_type):
        raise SchemaError("%s must be a %s" % (context, enum_type.__name__))
    return value


def _require_protocol_version(actual: Any, expected: str, context: str) -> str:
    if type(actual) is not str:
        raise SchemaError("%s must be a string" % context)
    if actual != expected:
        raise UnsupportedProtocolVersionError(expected=expected, actual=actual)
    return actual


def _portable_artifact_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _relative_artifact_path(value: Any, context: str = "artifact path") -> str:
    if type(value) is not str or not value or len(value) > MAX_ARTIFACT_PATH_CHARS:
        raise SchemaError("%s must be a bounded non-empty relative path" % context)
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise SchemaError("%s must contain valid Unicode" % context) from exc
    if value.startswith("/") or "\\" in value or "//" in value:
        raise SchemaError("%s must be normalized POSIX relative form" % context)
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SchemaError("%s may not traverse directories" % context)
    for part in parts:
        validate_path_segment(part, context)
    return value


def _runner_artifact_name(value: Any) -> str:
    """Validate one Runner-owned JSON metadata filename."""

    if (
        type(value) is not str
        or len(value) > MAX_RUNNER_ARTIFACT_NAME_CHARS
        or _RUNNER_ARTIFACT_NAME_RE.fullmatch(value) is None
        or value not in _KNOWN_RUNNER_ARTIFACT_NAMES
    ):
        raise SchemaError("unknown or unsafe Runner artifact name")
    return value


def _required_runner_artifact_names(submission: EvalSubmission) -> frozenset[str]:
    names = set(_CONTROL_RUNNER_ARTIFACT_NAMES)
    if submission.trace_ref is not None and submission.trace_ref.type is TraceType.LOCAL_PATH:
        names.update(_MANDATORY_EXECUTION_RUNNER_ARTIFACT_NAMES)
    return frozenset(names)


def _check_bounded_canonical_payload_size(
    value: Any,
    maximum: int,
    context: str,
) -> None:
    """Measure canonical JSON incrementally and stop at the byte boundary."""

    size = 0
    active_containers: set[int] = set()

    def add(amount: int) -> None:
        nonlocal size
        size += amount
        if size > maximum:
            raise SchemaError(
                "%s exceeds the canonical byte limit of %d"
                % (context, maximum)
            )

    def add_string(value: str) -> None:
        add(2)
        for offset in range(0, len(value), 4096):
            encoded = json.dumps(
                value[offset : offset + 4096],
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8", "strict")
            add(len(encoded) - 2)

    def visit(value: Any, depth: int) -> None:
        if depth > MAX_JSON_DEPTH:
            raise SchemaError("%s exceeds the maximum JSON depth" % context)
        if value is None:
            add(4)
            return
        if type(value) is bool:
            add(4 if value else 5)
            return
        if type(value) is int:
            bit_length = abs(value).bit_length()
            approximate_digits = (bit_length * 30103) // 100000 + 1
            if approximate_digits > MAX_JSON_INTEGER_DIGITS:
                raise SchemaError("%s contains an oversized integer" % context)
            add(len(str(value)))
            return
        if type(value) is float:
            add(
                len(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode("utf-8", "strict")
                )
            )
            return
        if type(value) is str:
            add_string(value)
            return
        if type(value) in (list, tuple):
            identity = id(value)
            if identity in active_containers:
                raise SchemaError("%s contains a circular value" % context)
            active_containers.add(identity)
            try:
                add(2)
                for index, item in enumerate(value):
                    if index:
                        add(1)
                    visit(item, depth + 1)
            finally:
                active_containers.remove(identity)
            return
        if type(value) is dict:
            identity = id(value)
            if identity in active_containers:
                raise SchemaError("%s contains a circular value" % context)
            active_containers.add(identity)
            try:
                add(2)
                for index, (key, item) in enumerate(value.items()):
                    if type(key) is not str:
                        raise SchemaError(
                            "%s contains a non-string object key" % context
                        )
                    if index:
                        add(1)
                    add_string(key)
                    add(1)
                    visit(item, depth + 1)
            finally:
                active_containers.remove(identity)
            return
        raise SchemaError("%s contains a non-JSON value" % context)

    try:
        visit(value, 0)
    except SchemaError:
        raise
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SchemaError("%s is not canonical JSON" % context) from exc


def _validate_materialization_path_coverage(
    file_paths: Iterable[str],
    readable_paths: Iterable[str],
) -> None:
    """Validate readable-path coverage with one indexed component walk."""

    files = _bounded_tuple(
        file_paths,
        "materialization file paths",
        MAX_AGENT_VISIBLE_FILES,
    )
    readable = _bounded_tuple(
        readable_paths,
        "materialization readable paths",
        MAX_AGENT_VISIBLE_FILES,
    )
    terminal = object()
    trie: Dict[Any, Any] = {}
    covered = [False] * len(readable)
    for index, path in enumerate(readable):
        node = trie
        for component in path.split("/"):
            child = node.get(component)
            if child is None:
                child = {}
                node[component] = child
            node = child
        node.setdefault(terminal, []).append(index)

    for path in files:
        node = trie
        authorized = False
        for component in path.split("/"):
            child = node.get(component)
            if child is None:
                break
            node = child
            for index in node.get(terminal, ()):
                covered[index] = True
                authorized = True
        if not authorized:
            raise SchemaError("materialization file is outside TargetAccess")

    if not all(covered):
        raise SchemaError(
            "TargetAccess path has no Agent-visible file binding"
        )


@dataclass(frozen=True)
class ArtifactRef(_JsonModel):
    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _relative_artifact_path(self.relative_path)
        _digest(self.sha256, "artifact.sha256")
        _integer(
            self.size_bytes,
            "artifact.size_bytes",
            minimum=0,
            maximum=MAX_COUNTER,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactRef":
        payload = _object(value, "artifact")
        _exact_fields(payload, ("relative_path", "sha256", "size_bytes"), "artifact")
        return cls(
            relative_path=_relative_artifact_path(payload["relative_path"]),
            sha256=_digest(payload["sha256"], "artifact.sha256"),
            size_bytes=_integer(
                payload["size_bytes"],
                "artifact.size_bytes",
                minimum=0,
                maximum=MAX_COUNTER,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class TargetAccess(_JsonModel):
    """The exact relative, read-only Target projection granted to an Agent."""

    target_materialization_id: str
    readable_relative_paths: Tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.target_materialization_id) is not str
            or _MATERIALIZATION_ID_RE.fullmatch(
                self.target_materialization_id
            )
            is None
        ):
            raise SchemaError(
                "target_access.target_materialization_id is invalid"
            )
        if type(self.readable_relative_paths) not in (tuple, list):
            raise SchemaError(
                "target_access.readable_relative_paths must be a list or tuple"
            )
        if not self.readable_relative_paths or (
            len(self.readable_relative_paths) > MAX_AGENT_VISIBLE_FILES
        ):
            raise SchemaError(
                "target_access.readable_relative_paths must contain between 1 and %d paths"
                % MAX_AGENT_VISIBLE_FILES
            )
        paths = tuple(
            _relative_artifact_path(
                item,
                "target_access.readable_relative_paths[%d]" % index,
            )
            for index, item in enumerate(self.readable_relative_paths)
        )
        keys = [_portable_artifact_key(item) for item in paths]
        if len(keys) != len(set(keys)):
            raise SchemaError(
                "target_access.readable_relative_paths contains a portable path collision"
            )
        object.__setattr__(self, "readable_relative_paths", tuple(sorted(paths)))

    @classmethod
    def from_dict(cls, value: Any) -> "TargetAccess":
        payload = _object(value, "TargetAccess")
        _exact_fields(
            payload,
            ("target_materialization_id", "readable_relative_paths"),
            "TargetAccess",
        )
        paths = _array(
            payload["readable_relative_paths"],
            "target_access.readable_relative_paths",
            MAX_AGENT_VISIBLE_FILES,
        )
        return cls(
            target_materialization_id=_identifier(
                payload["target_materialization_id"],
                "target_access.target_materialization_id",
            ),
            readable_relative_paths=tuple(paths),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_materialization_id": self.target_materialization_id,
            "readable_relative_paths": list(self.readable_relative_paths),
        }


@dataclass(frozen=True)
class AgentVisibleFileBinding(_JsonModel):
    """One content-addressed file exposed through ``TargetAccess``."""

    role: str
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _identifier(self.role, "agent-visible file.role")
        _relative_artifact_path(
            self.relative_path, "agent-visible file.relative_path"
        )
        _integer(
            self.size_bytes,
            "agent-visible file.size_bytes",
            minimum=0,
            maximum=MAX_COUNTER,
        )
        _digest(self.sha256, "agent-visible file.sha256")

    @classmethod
    def from_dict(cls, value: Any) -> "AgentVisibleFileBinding":
        payload = _object(value, "AgentVisibleFileBinding")
        _exact_fields(
            payload,
            ("role", "relative_path", "size_bytes", "sha256"),
            "AgentVisibleFileBinding",
        )
        return cls(
            role=_identifier(payload["role"], "agent-visible file.role"),
            relative_path=_relative_artifact_path(
                payload["relative_path"],
                "agent-visible file.relative_path",
            ),
            size_bytes=_integer(
                payload["size_bytes"],
                "agent-visible file.size_bytes",
                minimum=0,
                maximum=MAX_COUNTER,
            ),
            sha256=_digest(payload["sha256"], "agent-visible file.sha256"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


_MATERIALIZATION_ID_SIZE_PLACEHOLDER = "materialization-" + ("0" * 64)


def _preflight_trial_materialization_size(
    *,
    schema_version: str,
    run_id: str,
    task_id: str,
    trial_id: str,
    attempt: int,
    eval_input_digest: str,
    review_target_digest: str,
    wire_contract: WireContractV2,
    suite_preparation_binding_digest: Optional[str],
    prepared_source_id: str,
    adapter_capabilities_digest: str,
    readable_relative_paths: Tuple[str, ...],
    files: Tuple[AgentVisibleFileBinding, ...],
    replay_binding_digest: str,
    materialization_id: str,
) -> None:
    """Gate exact canonical bytes without sorting or deriving identity."""

    payload = {
        "schema_version": schema_version,
        "run_id": run_id,
        "task_id": task_id,
        "trial_id": trial_id,
        "attempt": attempt,
        "eval_input_digest": eval_input_digest,
        "review_target_digest": review_target_digest,
        "wire_contract": wire_contract.to_dict(),
        "suite_preparation_binding_digest": (
            suite_preparation_binding_digest
        ),
        "prepared_source_id": prepared_source_id,
        "adapter_capabilities_digest": adapter_capabilities_digest,
        "target_access": {
            "target_materialization_id": materialization_id,
            "readable_relative_paths": list(readable_relative_paths),
        },
        "files": [item.to_dict() for item in files],
        "replay_binding_digest": replay_binding_digest,
        "materialization_id": materialization_id,
    }
    _check_bounded_canonical_payload_size(
        payload,
        MAX_TRIAL_MATERIALIZATION_BYTES,
        "TrialMaterializationManifest",
    )


@dataclass(frozen=True)
class TrialMaterializationManifest(_JsonModel):
    """Immutable per-attempt Target materialization identity and access grant."""

    SCHEMA_VERSION: ClassVar[str] = EVAL_TRIAL_MATERIALIZATION_SCHEMA_VERSION

    schema_version: str
    run_id: str
    task_id: str
    trial_id: str
    attempt: int
    eval_input_digest: str
    review_target_digest: str
    wire_contract: WireContractV2
    suite_preparation_binding_digest: Optional[str]
    prepared_source_id: str
    adapter_capabilities_digest: str
    target_access: TargetAccess
    files: Tuple[AgentVisibleFileBinding, ...]
    replay_binding_digest: str
    materialization_id: str

    def __post_init__(self) -> None:
        _require_protocol_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "TrialMaterializationManifest.schema_version",
        )
        validate_run_id(self.run_id)
        _identifier(self.task_id, "materialization.task_id")
        validate_trial_id_shape(self.trial_id)
        _integer(
            self.attempt,
            "materialization.attempt",
            minimum=1,
            maximum=MAX_TRIAL_ATTEMPTS,
        )
        _digest(self.eval_input_digest, "materialization.eval_input_digest")
        _digest(
            self.review_target_digest,
            "materialization.review_target_digest",
        )
        if not isinstance(self.wire_contract, WireContractV2):
            raise SchemaError(
                "materialization.wire_contract must be a WireContractV2"
            )
        if self.suite_preparation_binding_digest is not None:
            _digest(
                self.suite_preparation_binding_digest,
                "materialization.suite_preparation_binding_digest",
            )
        _identifier(self.prepared_source_id, "materialization.prepared_source_id")
        _digest(
            self.adapter_capabilities_digest,
            "materialization.adapter_capabilities_digest",
        )
        if not isinstance(self.target_access, TargetAccess):
            raise SchemaError(
                "materialization.target_access must be a TargetAccess"
            )
        if type(self.files) not in (tuple, list):
            raise SchemaError("materialization.files must be a list or tuple")
        if not self.files or len(self.files) > MAX_AGENT_VISIBLE_FILES:
            raise SchemaError(
                "materialization.files must contain between 1 and %d entries"
                % MAX_AGENT_VISIBLE_FILES
            )
        files = tuple(self.files)
        if any(not isinstance(item, AgentVisibleFileBinding) for item in files):
            raise SchemaError(
                "materialization.files must contain AgentVisibleFileBinding values"
            )
        file_keys = [_portable_artifact_key(item.relative_path) for item in files]
        if len(file_keys) != len(set(file_keys)):
            raise SchemaError(
                "materialization.files contains a portable path collision"
            )
        readable = self.target_access.readable_relative_paths
        _digest(
            self.replay_binding_digest,
            "materialization.replay_binding_digest",
        )
        if (
            type(self.materialization_id) is not str
            or _MATERIALIZATION_ID_RE.fullmatch(self.materialization_id) is None
        ):
            raise SchemaError("materialization.materialization_id is invalid")
        if (
            self.target_access.target_materialization_id
            != self.materialization_id
        ):
            raise SchemaError(
                "TargetAccess target_materialization_id does not match materialization"
            )
        _preflight_trial_materialization_size(
            schema_version=self.schema_version,
            run_id=self.run_id,
            task_id=self.task_id,
            trial_id=self.trial_id,
            attempt=self.attempt,
            eval_input_digest=self.eval_input_digest,
            review_target_digest=self.review_target_digest,
            wire_contract=self.wire_contract,
            suite_preparation_binding_digest=(
                self.suite_preparation_binding_digest
            ),
            prepared_source_id=self.prepared_source_id,
            adapter_capabilities_digest=self.adapter_capabilities_digest,
            readable_relative_paths=readable,
            files=files,
            replay_binding_digest=self.replay_binding_digest,
            materialization_id=self.materialization_id,
        )
        _validate_materialization_path_coverage(
            (item.relative_path for item in files),
            readable,
        )
        ordered_files = tuple(sorted(files, key=lambda item: item.relative_path))
        object.__setattr__(self, "files", ordered_files)
        expected_id = self.derive_materialization_id(
            run_id=self.run_id,
            task_id=self.task_id,
            trial_id=self.trial_id,
            attempt=self.attempt,
            eval_input_digest=self.eval_input_digest,
            review_target_digest=self.review_target_digest,
            wire_contract=self.wire_contract,
            suite_preparation_binding_digest=(
                self.suite_preparation_binding_digest
            ),
            prepared_source_id=self.prepared_source_id,
            adapter_capabilities_digest=self.adapter_capabilities_digest,
            readable_relative_paths=self.target_access.readable_relative_paths,
            files=ordered_files,
            replay_binding_digest=self.replay_binding_digest,
        )
        if self.materialization_id != expected_id:
            raise SchemaError(
                "materialization_id does not match its canonical identity"
            )
        validate_safe_json(self.to_dict(), "materialization")

    @classmethod
    def derive_materialization_id(
        cls,
        *,
        run_id: str,
        task_id: str,
        trial_id: str,
        attempt: int,
        eval_input_digest: str,
        review_target_digest: str,
        wire_contract: WireContractV2,
        suite_preparation_binding_digest: Optional[str],
        prepared_source_id: str,
        adapter_capabilities_digest: str,
        readable_relative_paths: Iterable[str],
        files: Iterable[AgentVisibleFileBinding],
        replay_binding_digest: str,
    ) -> str:
        if not isinstance(wire_contract, WireContractV2):
            raise SchemaError(
                "materialization.wire_contract must be a WireContractV2"
            )
        raw_paths = _bounded_tuple(
            readable_relative_paths,
            "materialization.readable_relative_paths",
            MAX_AGENT_VISIBLE_FILES,
        )
        paths = tuple(
            _relative_artifact_path(
                item,
                "materialization.readable_relative_paths[%d]" % index,
            )
            for index, item in enumerate(raw_paths)
        )
        raw_files = _bounded_tuple(
            files,
            "materialization.files",
            MAX_AGENT_VISIBLE_FILES,
        )
        if any(
            not isinstance(item, AgentVisibleFileBinding)
            for item in raw_files
        ):
            raise SchemaError(
                "materialization.files must contain AgentVisibleFileBinding values"
            )
        _preflight_trial_materialization_size(
            schema_version=cls.SCHEMA_VERSION,
            run_id=run_id,
            task_id=task_id,
            trial_id=trial_id,
            attempt=attempt,
            eval_input_digest=eval_input_digest,
            review_target_digest=review_target_digest,
            wire_contract=wire_contract,
            suite_preparation_binding_digest=suite_preparation_binding_digest,
            prepared_source_id=prepared_source_id,
            adapter_capabilities_digest=adapter_capabilities_digest,
            readable_relative_paths=paths,
            files=raw_files,
            replay_binding_digest=replay_binding_digest,
            materialization_id=_MATERIALIZATION_ID_SIZE_PLACEHOLDER,
        )
        paths = tuple(sorted(paths))
        ordered_files = tuple(
            sorted(raw_files, key=lambda item: item.relative_path)
        )
        return stable_id(
            "materialization",
            {
                "schema_version": cls.SCHEMA_VERSION,
                "run_id": run_id,
                "task_id": task_id,
                "trial_id": trial_id,
                "attempt": attempt,
                "eval_input_digest": eval_input_digest,
                "review_target_digest": review_target_digest,
                "wire_contract": wire_contract.to_dict(),
                "suite_preparation_binding_digest": (
                    suite_preparation_binding_digest
                ),
                "prepared_source_id": prepared_source_id,
                "adapter_capabilities_digest": adapter_capabilities_digest,
                "readable_relative_paths": list(paths),
                "files": [item.to_dict() for item in ordered_files],
                "replay_binding_digest": replay_binding_digest,
            },
        )

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        task_id: str,
        trial_id: str,
        attempt: int,
        eval_input_digest: str,
        review_target_digest: str,
        wire_contract: WireContractV2,
        suite_preparation_binding_digest: Optional[str],
        prepared_source_id: str,
        adapter_capabilities_digest: str,
        readable_relative_paths: Iterable[str],
        files: Iterable[AgentVisibleFileBinding],
        replay_binding_digest: str,
    ) -> "TrialMaterializationManifest":
        raw_paths = _bounded_tuple(
            readable_relative_paths,
            "materialization.readable_relative_paths",
            MAX_AGENT_VISIBLE_FILES,
        )
        paths = tuple(
            _relative_artifact_path(
                item, "materialization.readable_relative_paths"
            )
            for item in raw_paths
        )
        file_bindings = _bounded_tuple(
            files,
            "materialization.files",
            MAX_AGENT_VISIBLE_FILES,
        )
        if any(
            not isinstance(item, AgentVisibleFileBinding)
            for item in file_bindings
        ):
            raise SchemaError(
                "materialization.files must contain AgentVisibleFileBinding values"
            )
        if not isinstance(wire_contract, WireContractV2):
            raise SchemaError(
                "materialization.wire_contract must be a WireContractV2"
            )
        _preflight_trial_materialization_size(
            schema_version=cls.SCHEMA_VERSION,
            run_id=run_id,
            task_id=task_id,
            trial_id=trial_id,
            attempt=attempt,
            eval_input_digest=eval_input_digest,
            review_target_digest=review_target_digest,
            wire_contract=wire_contract,
            suite_preparation_binding_digest=suite_preparation_binding_digest,
            prepared_source_id=prepared_source_id,
            adapter_capabilities_digest=adapter_capabilities_digest,
            readable_relative_paths=paths,
            files=file_bindings,
            replay_binding_digest=replay_binding_digest,
            materialization_id=_MATERIALIZATION_ID_SIZE_PLACEHOLDER,
        )
        materialization_id = cls.derive_materialization_id(
            run_id=run_id,
            task_id=task_id,
            trial_id=trial_id,
            attempt=attempt,
            eval_input_digest=eval_input_digest,
            review_target_digest=review_target_digest,
            wire_contract=wire_contract,
            suite_preparation_binding_digest=suite_preparation_binding_digest,
            prepared_source_id=prepared_source_id,
            adapter_capabilities_digest=adapter_capabilities_digest,
            readable_relative_paths=paths,
            files=file_bindings,
            replay_binding_digest=replay_binding_digest,
        )
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            run_id=run_id,
            task_id=task_id,
            trial_id=trial_id,
            attempt=attempt,
            eval_input_digest=eval_input_digest,
            review_target_digest=review_target_digest,
            wire_contract=wire_contract,
            suite_preparation_binding_digest=suite_preparation_binding_digest,
            prepared_source_id=prepared_source_id,
            adapter_capabilities_digest=adapter_capabilities_digest,
            target_access=TargetAccess(
                target_materialization_id=materialization_id,
                readable_relative_paths=paths,
            ),
            files=file_bindings,
            replay_binding_digest=replay_binding_digest,
            materialization_id=materialization_id,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "TrialMaterializationManifest":
        payload = _object(value, "TrialMaterializationManifest")
        if "schema_version" in payload:
            _require_protocol_version(
                payload["schema_version"],
                cls.SCHEMA_VERSION,
                "TrialMaterializationManifest.schema_version",
            )
        _exact_fields(
            payload,
            (
                "schema_version",
                "run_id",
                "task_id",
                "trial_id",
                "attempt",
                "eval_input_digest",
                "review_target_digest",
                "wire_contract",
                "suite_preparation_binding_digest",
                "prepared_source_id",
                "adapter_capabilities_digest",
                "target_access",
                "files",
                "replay_binding_digest",
                "materialization_id",
            ),
            "TrialMaterializationManifest",
        )
        _check_bounded_canonical_payload_size(
            payload,
            MAX_TRIAL_MATERIALIZATION_BYTES,
            "TrialMaterializationManifest",
        )
        wire_contract = WireContractV2.from_dict(payload["wire_contract"])
        raw_files = _array(
            payload["files"],
            "materialization.files",
            MAX_AGENT_VISIBLE_FILES,
        )
        return cls(
            schema_version=payload["schema_version"],
            run_id=validate_run_id(payload["run_id"]),
            task_id=_identifier(payload["task_id"], "materialization.task_id"),
            trial_id=validate_trial_id_shape(payload["trial_id"]),
            attempt=_integer(
                payload["attempt"],
                "materialization.attempt",
                minimum=1,
                maximum=MAX_TRIAL_ATTEMPTS,
            ),
            eval_input_digest=_digest(
                payload["eval_input_digest"],
                "materialization.eval_input_digest",
            ),
            review_target_digest=_digest(
                payload["review_target_digest"],
                "materialization.review_target_digest",
            ),
            wire_contract=wire_contract,
            suite_preparation_binding_digest=(
                None
                if payload["suite_preparation_binding_digest"] is None
                else _digest(
                    payload["suite_preparation_binding_digest"],
                    "materialization.suite_preparation_binding_digest",
                )
            ),
            prepared_source_id=_identifier(
                payload["prepared_source_id"],
                "materialization.prepared_source_id",
            ),
            adapter_capabilities_digest=_digest(
                payload["adapter_capabilities_digest"],
                "materialization.adapter_capabilities_digest",
            ),
            target_access=TargetAccess.from_dict(payload["target_access"]),
            files=tuple(
                AgentVisibleFileBinding.from_dict(item) for item in raw_files
            ),
            replay_binding_digest=_digest(
                payload["replay_binding_digest"],
                "materialization.replay_binding_digest",
            ),
            materialization_id=_identifier(
                payload["materialization_id"],
                "materialization.materialization_id",
            ),
        )

    @classmethod
    def from_json(cls, data: Any) -> "TrialMaterializationManifest":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_TRIAL_MATERIALIZATION_BYTES,
                "TrialMaterializationManifest JSON",
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "trial_id": self.trial_id,
            "attempt": self.attempt,
            "eval_input_digest": self.eval_input_digest,
            "review_target_digest": self.review_target_digest,
            "wire_contract": self.wire_contract.to_dict(),
            "suite_preparation_binding_digest": (
                self.suite_preparation_binding_digest
            ),
            "prepared_source_id": self.prepared_source_id,
            "adapter_capabilities_digest": self.adapter_capabilities_digest,
            "target_access": self.target_access.to_dict(),
            "files": [item.to_dict() for item in self.files],
            "replay_binding_digest": self.replay_binding_digest,
            "materialization_id": self.materialization_id,
        }


@dataclass(frozen=True)
class TrialManifest(_JsonModel):
    """Immutable Trial plan; it deliberately contains no execution status."""

    SCHEMA_VERSION: ClassVar[str] = EVAL_TRIAL_MANIFEST_SCHEMA_VERSION

    schema_version: str
    run_id: str
    task_id: str
    case_path_id: str
    canonical_case_digest: str
    eval_input_digest: str
    wire_contract: WireContractV2
    target_kind: ReviewTargetKind
    materializer_protocol: str
    suite_preparation_binding_digest: Optional[str]
    adapter_capabilities_digest: str
    trial_id: str
    trial_index: int
    seed: int
    agent_config_digest: str
    initial_evaluator_execution_digest: str

    def __post_init__(self) -> None:
        _require_protocol_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "TrialManifest.schema_version",
        )
        validate_run_id(self.run_id)
        _identifier(self.task_id, "trial_manifest.task_id")
        validate_case_path_id(self.case_path_id)
        if self.case_path_id != derive_case_path_id(self.task_id):
            raise SchemaError("case_path_id does not match opaque task_id")
        _digest(
            self.canonical_case_digest,
            "trial_manifest.canonical_case_digest",
        )
        _digest(self.eval_input_digest, "trial_manifest.eval_input_digest")
        if not isinstance(self.wire_contract, WireContractV2):
            raise SchemaError(
                "trial_manifest.wire_contract must be a WireContractV2"
            )
        _require_enum(
            ReviewTargetKind,
            self.target_kind,
            "trial_manifest.target_kind",
        )
        if self.target_kind is not self.wire_contract.review_target_kind:
            raise SchemaError(
                "trial_manifest.target_kind does not match wire contract"
            )
        _identifier(
            self.materializer_protocol,
            "trial_manifest.materializer_protocol",
        )
        if self.materializer_protocol != self.wire_contract.materializer_protocol:
            raise SchemaError(
                "trial_manifest.materializer_protocol does not match wire contract"
            )
        if self.suite_preparation_binding_digest is not None:
            _digest(
                self.suite_preparation_binding_digest,
                "trial_manifest.suite_preparation_binding_digest",
            )
        _digest(
            self.adapter_capabilities_digest,
            "trial_manifest.adapter_capabilities_digest",
        )
        validate_trial_id(self.trial_id, self.run_id, self.task_id, self.trial_index)
        _integer(
            self.trial_index,
            "trial_manifest.trial_index",
            minimum=1,
            maximum=MAX_COUNTER,
        )
        if self.seed != derive_trial_seed(self.run_id, self.task_id, self.trial_index):
            raise SchemaError("Trial seed does not match its canonical identity")
        _digest(self.agent_config_digest, "trial_manifest.agent_config_digest")
        _digest(
            self.initial_evaluator_execution_digest,
            "trial_manifest.initial_evaluator_execution_digest",
        )
        validate_safe_json(self.to_dict(), "trial_manifest")
        _check_model_size(self, MAX_TRIAL_MANIFEST_BYTES, "TrialManifest")

    @classmethod
    def from_dict(cls, value: Any) -> "TrialManifest":
        payload = _object(value, "TrialManifest")
        if "schema_version" in payload:
            _require_protocol_version(
                payload["schema_version"],
                cls.SCHEMA_VERSION,
                "TrialManifest.schema_version",
            )
        _exact_fields(
            payload,
            (
                "schema_version",
                "run_id",
                "task_id",
                "case_path_id",
                "canonical_case_digest",
                "eval_input_digest",
                "wire_contract",
                "target_kind",
                "materializer_protocol",
                "suite_preparation_binding_digest",
                "adapter_capabilities_digest",
                "trial_id",
                "trial_index",
                "seed",
                "agent_config_digest",
                "initial_evaluator_execution_digest",
            ),
            "TrialManifest",
        )
        wire_contract = WireContractV2.from_dict(payload["wire_contract"])
        return cls(
            schema_version=payload["schema_version"],
            run_id=validate_run_id(payload["run_id"]),
            task_id=_identifier(payload["task_id"], "trial_manifest.task_id"),
            case_path_id=validate_case_path_id(payload["case_path_id"]),
            canonical_case_digest=_digest(
                payload["canonical_case_digest"],
                "trial_manifest.canonical_case_digest",
            ),
            eval_input_digest=_digest(
                payload["eval_input_digest"], "trial_manifest.eval_input_digest"
            ),
            wire_contract=wire_contract,
            target_kind=_enum_value(
                ReviewTargetKind,
                payload["target_kind"],
                "trial_manifest.target_kind",
            ),
            materializer_protocol=_identifier(
                payload["materializer_protocol"],
                "trial_manifest.materializer_protocol",
            ),
            suite_preparation_binding_digest=(
                None
                if payload["suite_preparation_binding_digest"] is None
                else _digest(
                    payload["suite_preparation_binding_digest"],
                    "trial_manifest.suite_preparation_binding_digest",
                )
            ),
            adapter_capabilities_digest=_digest(
                payload["adapter_capabilities_digest"],
                "trial_manifest.adapter_capabilities_digest",
            ),
            trial_id=validate_trial_id_shape(payload["trial_id"]),
            trial_index=_integer(
                payload["trial_index"],
                "trial_manifest.trial_index",
                minimum=1,
                maximum=MAX_COUNTER,
            ),
            seed=_integer(
                payload["seed"],
                "trial_manifest.seed",
                minimum=0,
                maximum=(1 << 63) - 1,
            ),
            agent_config_digest=_digest(
                payload["agent_config_digest"],
                "trial_manifest.agent_config_digest",
            ),
            initial_evaluator_execution_digest=_digest(
                payload["initial_evaluator_execution_digest"],
                "trial_manifest.initial_evaluator_execution_digest",
            ),
        )

    @classmethod
    def from_json(cls, data: Any) -> "TrialManifest":
        return cls.from_dict(
            _strict_json_loads(data, MAX_TRIAL_MANIFEST_BYTES, "TrialManifest JSON")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "case_path_id": self.case_path_id,
            "canonical_case_digest": self.canonical_case_digest,
            "eval_input_digest": self.eval_input_digest,
            "wire_contract": self.wire_contract.to_dict(),
            "target_kind": self.target_kind.value,
            "materializer_protocol": self.materializer_protocol,
            "suite_preparation_binding_digest": (
                self.suite_preparation_binding_digest
            ),
            "adapter_capabilities_digest": self.adapter_capabilities_digest,
            "trial_id": self.trial_id,
            "trial_index": self.trial_index,
            "seed": self.seed,
            "agent_config_digest": self.agent_config_digest,
            "initial_evaluator_execution_digest": self.initial_evaluator_execution_digest,
        }


@dataclass(frozen=True)
class RunTrialPlan(_JsonModel):
    task_id: str
    case_path_id: str
    canonical_case_digest: str
    eval_input_digest: str
    trial_id: str
    trial_index: int
    manifest: ArtifactRef

    def __post_init__(self) -> None:
        _identifier(self.task_id, "run trial.task_id")
        validate_case_path_id(self.case_path_id)
        if self.case_path_id != derive_case_path_id(self.task_id):
            raise SchemaError("run trial.case_path_id does not match task_id")
        _digest(
            self.canonical_case_digest,
            "run trial.canonical_case_digest",
        )
        _digest(self.eval_input_digest, "run trial.eval_input_digest")
        validate_trial_id_shape(self.trial_id)
        _integer(
            self.trial_index,
            "run trial.trial_index",
            minimum=1,
            maximum=MAX_COUNTER,
        )
        if not isinstance(self.manifest, ArtifactRef):
            raise SchemaError("run trial.manifest must be an ArtifactRef")
        expected = "cases/%s/trials/%s/trial_manifest.json" % (
            self.case_path_id,
            self.trial_id,
        )
        if self.manifest.relative_path != expected:
            raise SchemaError("run trial.manifest references the wrong immutable plan")

    @classmethod
    def from_dict(cls, value: Any) -> "RunTrialPlan":
        payload = _object(value, "run trial")
        _exact_fields(
            payload,
            (
                "task_id",
                "case_path_id",
                "canonical_case_digest",
                "eval_input_digest",
                "trial_id",
                "trial_index",
                "manifest",
            ),
            "run trial",
        )
        return cls(
            task_id=_identifier(payload["task_id"], "run trial.task_id"),
            case_path_id=validate_case_path_id(payload["case_path_id"]),
            canonical_case_digest=_digest(
                payload["canonical_case_digest"],
                "run trial.canonical_case_digest",
            ),
            eval_input_digest=_digest(
                payload["eval_input_digest"], "run trial.eval_input_digest"
            ),
            trial_id=validate_trial_id_shape(payload["trial_id"]),
            trial_index=_integer(
                payload["trial_index"],
                "run trial.trial_index",
                minimum=1,
                maximum=MAX_COUNTER,
            ),
            manifest=ArtifactRef.from_dict(payload["manifest"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "case_path_id": self.case_path_id,
            "canonical_case_digest": self.canonical_case_digest,
            "eval_input_digest": self.eval_input_digest,
            "trial_id": self.trial_id,
            "trial_index": self.trial_index,
            "manifest": self.manifest.to_dict(),
        }


@dataclass(frozen=True)
class RunManifest(_JsonModel):
    """Immutable run plan; state is derived from Trial receipts."""

    SCHEMA_VERSION: ClassVar[str] = EVAL_RUN_MANIFEST_SCHEMA_VERSION

    schema_version: str
    run_id: str
    run_config: ArtifactRef
    case_snapshot: ArtifactRef
    wire_contract: WireContractV2
    suite_preparation_binding_digest: Optional[str]
    adapter_capabilities_digest: str
    agent_config_digest: str
    initial_evaluator_execution_digest: str
    trials: Tuple[RunTrialPlan, ...]

    def __post_init__(self) -> None:
        _require_protocol_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "RunManifest.schema_version",
        )
        validate_run_id(self.run_id)
        if not isinstance(self.run_config, ArtifactRef):
            raise SchemaError("run_manifest.run_config must be an ArtifactRef")
        if self.run_config.relative_path != "run_config.json":
            raise SchemaError("run_manifest.run_config must reference run_config.json")
        if not isinstance(self.case_snapshot, ArtifactRef):
            raise SchemaError(
                "run_manifest.case_snapshot must be an ArtifactRef"
            )
        if self.case_snapshot.relative_path != "case_snapshot.json":
            raise SchemaError(
                "run_manifest.case_snapshot must reference case_snapshot.json"
            )
        if not isinstance(self.wire_contract, WireContractV2):
            raise SchemaError(
                "run_manifest.wire_contract must be a WireContractV2"
            )
        if self.suite_preparation_binding_digest is not None:
            _digest(
                self.suite_preparation_binding_digest,
                "run_manifest.suite_preparation_binding_digest",
            )
        _digest(
            self.adapter_capabilities_digest,
            "run_manifest.adapter_capabilities_digest",
        )
        _digest(self.agent_config_digest, "run_manifest.agent_config_digest")
        _digest(
            self.initial_evaluator_execution_digest,
            "run_manifest.initial_evaluator_execution_digest",
        )
        if type(self.trials) not in (list, tuple):
            raise SchemaError("run_manifest.trials must be a list or tuple")
        if not self.trials or len(self.trials) > MAX_MANIFEST_TRIALS:
            raise SchemaError(
                "run_manifest.trials must contain between 1 and %d entries"
                % MAX_MANIFEST_TRIALS
            )
        trials = tuple(self.trials)
        if any(not isinstance(item, RunTrialPlan) for item in trials):
            raise SchemaError("run_manifest.trials must contain RunTrialPlan values")
        for item in trials:
            validate_trial_id(
                item.trial_id,
                self.run_id,
                item.task_id,
                item.trial_index,
            )
        identities = [(item.task_id, item.trial_index) for item in trials]
        if len(identities) != len(set(identities)):
            raise SchemaError("run_manifest.trials contains duplicate plans")
        if len({item.trial_id for item in trials}) != len(trials):
            raise SchemaError("run_manifest.trials contains duplicate trial_id values")
        if len({item.manifest.relative_path for item in trials}) != len(trials):
            raise SchemaError("run_manifest.trials contains duplicate manifest paths")
        object.__setattr__(
            self,
            "trials",
            tuple(sorted(trials, key=lambda item: (item.task_id, item.trial_index))),
        )
        validate_safe_json(self.to_dict(), "run_manifest")
        _check_model_size(self, MAX_RUN_MANIFEST_BYTES, "RunManifest")

    @classmethod
    def from_dict(cls, value: Any) -> "RunManifest":
        payload = _object(value, "RunManifest")
        if "schema_version" in payload:
            _require_protocol_version(
                payload["schema_version"],
                cls.SCHEMA_VERSION,
                "RunManifest.schema_version",
            )
        _exact_fields(
            payload,
            (
                "schema_version",
                "run_id",
                "run_config",
                "case_snapshot",
                "wire_contract",
                "suite_preparation_binding_digest",
                "adapter_capabilities_digest",
                "agent_config_digest",
                "initial_evaluator_execution_digest",
                "trials",
            ),
            "RunManifest",
        )
        wire_contract = WireContractV2.from_dict(payload["wire_contract"])
        trials = _array(payload["trials"], "run_manifest.trials", MAX_MANIFEST_TRIALS)
        if not trials:
            raise SchemaError("run_manifest.trials must not be empty")
        return cls(
            schema_version=payload["schema_version"],
            run_id=validate_run_id(payload["run_id"]),
            run_config=ArtifactRef.from_dict(payload["run_config"]),
            case_snapshot=ArtifactRef.from_dict(payload["case_snapshot"]),
            wire_contract=wire_contract,
            suite_preparation_binding_digest=(
                None
                if payload["suite_preparation_binding_digest"] is None
                else _digest(
                    payload["suite_preparation_binding_digest"],
                    "run_manifest.suite_preparation_binding_digest",
                )
            ),
            adapter_capabilities_digest=_digest(
                payload["adapter_capabilities_digest"],
                "run_manifest.adapter_capabilities_digest",
            ),
            agent_config_digest=_digest(
                payload["agent_config_digest"], "run_manifest.agent_config_digest"
            ),
            initial_evaluator_execution_digest=_digest(
                payload["initial_evaluator_execution_digest"],
                "run_manifest.initial_evaluator_execution_digest",
            ),
            trials=tuple(RunTrialPlan.from_dict(item) for item in trials),
        )

    @classmethod
    def from_json(cls, data: Any) -> "RunManifest":
        return cls.from_dict(
            _strict_json_loads(data, MAX_RUN_MANIFEST_BYTES, "RunManifest JSON")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_config": self.run_config.to_dict(),
            "case_snapshot": self.case_snapshot.to_dict(),
            "wire_contract": self.wire_contract.to_dict(),
            "suite_preparation_binding_digest": (
                self.suite_preparation_binding_digest
            ),
            "adapter_capabilities_digest": self.adapter_capabilities_digest,
            "agent_config_digest": self.agent_config_digest,
            "initial_evaluator_execution_digest": self.initial_evaluator_execution_digest,
            "trials": [item.to_dict() for item in self.trials],
        }


def _validate_failure(status: Optional[TrialStatus], code: Optional[FailureCode]) -> None:
    if status is None:
        if code is not None:
            raise SchemaError("nonterminal receipt requires failure_code=null")
    elif status is TrialStatus.COMPLETED:
        if code is not None:
            raise SchemaError("completed receipt requires failure_code=null")
    elif status not in _TERMINAL_STATUSES:
        raise SchemaError("terminal receipt has a nonterminal Trial status")
    elif code is None:
        raise SchemaError("terminal failure receipt requires failure_code")
    else:
        try:
            expected = TrialStatus(submission_status_for_failure(code).value)
        except (TypeError, ValueError) as exc:
            raise SchemaError("terminal receipt has an invalid failure_code") from exc
        if status is not expected:
            raise SchemaError(
                "%s receipt has an invalid failure_code" % status.value
            )


def derive_receipt_id(
    run_id: str,
    task_id: str,
    trial_id: str,
    stage: StageName,
    config_digest: str,
    *,
    attempt: Optional[int],
    evaluation_id: Optional[str],
    evaluation_revision: Optional[str],
) -> str:
    validate_run_id(run_id)
    _identifier(task_id, "receipt.task_id")
    validate_trial_id_shape(trial_id)
    _require_enum(StageName, stage, "receipt.stage")
    _digest(config_digest, "receipt.config_digest")
    if attempt is not None:
        _integer(attempt, "receipt.attempt", minimum=1, maximum=MAX_TRIAL_ATTEMPTS)
    if evaluation_id is not None:
        validate_evaluation_id_shape(evaluation_id)
    if evaluation_revision is not None:
        validate_path_segment(
            evaluation_revision, "receipt.evaluation_revision"
        )
    return stable_id(
        "receipt",
        run_id,
        task_id,
        trial_id,
        stage.value,
        config_digest,
        attempt,
        evaluation_id,
        evaluation_revision,
    )


def derive_pre_materialization_failure_binding(
    *,
    run_id: str,
    task_id: str,
    trial_id: str,
    attempt: int,
    eval_input_digest: str,
    review_target_digest: str,
) -> str:
    """Derive the sole target binding allowed before a Prepare commit."""

    validate_run_id(run_id)
    _identifier(task_id, "pre-materialization failure.task_id")
    validate_trial_id_shape(trial_id)
    normalized_attempt = _integer(
        attempt,
        "pre-materialization failure.attempt",
        minimum=1,
        maximum=MAX_TRIAL_ATTEMPTS,
    )
    _digest(
        eval_input_digest,
        "pre-materialization failure.eval_input_digest",
    )
    _digest(
        review_target_digest,
        "pre-materialization failure.review_target_digest",
    )
    return stable_id(
        "pre-materialization-failure",
        {
            "schema_version": PRE_MATERIALIZATION_FAILURE_BINDING_VERSION,
            "run_id": run_id,
            "task_id": task_id,
            "trial_id": trial_id,
            "attempt": normalized_attempt,
            "eval_input_digest": eval_input_digest,
            "review_target_digest": review_target_digest,
        },
    )


@dataclass(frozen=True)
class StageReceipt(_JsonModel):
    SCHEMA_VERSION: ClassVar[str] = EVAL_STAGE_RECEIPT_SCHEMA_VERSION

    schema_version: str
    receipt_id: str
    run_id: str
    task_id: str
    trial_id: str
    stage: StageName
    config_digest: str
    attempt: Optional[int]
    evaluation_id: Optional[str]
    evaluation_revision: Optional[str]
    artifacts: Tuple[ArtifactRef, ...]
    materialization_manifest: Optional[ArtifactRef]
    materialization_manifest_digest: Optional[str]
    materialization_id: Optional[str]
    eval_input_digest: Optional[str]
    review_target_digest: Optional[str]
    prepared_source_id: Optional[str]
    agent_visible_files: Tuple[AgentVisibleFileBinding, ...]
    adapter_capabilities_digest: Optional[str]
    target_access: Optional[TargetAccess]
    terminal_status: Optional[TrialStatus]
    failure_code: Optional[FailureCode]

    def __post_init__(self) -> None:
        _require_protocol_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "StageReceipt.schema_version",
        )
        validate_run_id(self.run_id)
        _identifier(self.task_id, "receipt.task_id")
        validate_trial_id_shape(self.trial_id)
        _require_enum(StageName, self.stage, "receipt.stage")
        _digest(self.config_digest, "receipt.config_digest")
        if self.attempt is not None:
            _integer(
                self.attempt,
                "receipt.attempt",
                minimum=1,
                maximum=MAX_TRIAL_ATTEMPTS,
            )
        if self.evaluation_id is not None:
            validate_evaluation_id_shape(self.evaluation_id)
        if self.evaluation_revision is not None:
            validate_path_segment(
                self.evaluation_revision, "receipt.evaluation_revision"
            )
        expected = derive_receipt_id(
            self.run_id,
            self.task_id,
            self.trial_id,
            self.stage,
            self.config_digest,
            attempt=self.attempt,
            evaluation_id=self.evaluation_id,
            evaluation_revision=self.evaluation_revision,
        )
        if self.receipt_id != expected:
            raise SchemaError("receipt_id does not match its canonical identity")
        if type(self.artifacts) not in (list, tuple):
            raise SchemaError("receipt.artifacts must be a list or tuple")
        if len(self.artifacts) > MAX_RECEIPT_ARTIFACTS:
            raise SchemaError("receipt.artifacts exceeds its item limit")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ArtifactRef) for item in artifacts):
            raise SchemaError("receipt.artifacts must contain ArtifactRef values")
        paths = [item.relative_path for item in artifacts]
        if len(paths) != len(set(paths)):
            raise SchemaError("receipt.artifacts contains duplicate paths")
        if self.materialization_manifest is not None and not isinstance(
            self.materialization_manifest, ArtifactRef
        ):
            raise SchemaError(
                "receipt.materialization_manifest must be an ArtifactRef or null"
            )
        if self.materialization_manifest_digest is not None:
            _digest(
                self.materialization_manifest_digest,
                "receipt.materialization_manifest_digest",
            )
        if self.materialization_id is not None and (
            type(self.materialization_id) is not str
            or _MATERIALIZATION_ID_RE.fullmatch(self.materialization_id) is None
        ):
            raise SchemaError("receipt.materialization_id is invalid")
        for field_name in (
            "eval_input_digest",
            "review_target_digest",
            "adapter_capabilities_digest",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                _digest(field_value, "receipt.%s" % field_name)
        if self.prepared_source_id is not None:
            _identifier(
                self.prepared_source_id, "receipt.prepared_source_id"
            )
        if type(self.agent_visible_files) not in (tuple, list):
            raise SchemaError(
                "receipt.agent_visible_files must be a list or tuple"
            )
        visible_files = tuple(self.agent_visible_files)
        if len(visible_files) > MAX_AGENT_VISIBLE_FILES or any(
            not isinstance(item, AgentVisibleFileBinding)
            for item in visible_files
        ):
            raise SchemaError(
                "receipt.agent_visible_files contains invalid bindings"
            )
        if self.target_access is not None and not isinstance(
            self.target_access, TargetAccess
        ):
            raise SchemaError(
                "receipt.target_access must be a TargetAccess or null"
            )
        if self.terminal_status is not None:
            _require_enum(TrialStatus, self.terminal_status, "receipt.terminal_status")
        if self.failure_code is not None:
            _require_enum(FailureCode, self.failure_code, "receipt.failure_code")
        _validate_failure(self.terminal_status, self.failure_code)

        if self.stage in {StageName.START, StageName.INCOMPLETE}:
            if (
                self.attempt is None
                or self.evaluation_id is not None
                or self.evaluation_revision is not None
                or artifacts
                or self.terminal_status is not None
            ):
                raise SchemaError("lifecycle receipt has an invalid field combination")
        elif self.stage is StageName.PREPARE:
            if (
                self.attempt is None
                or self.evaluation_id is not None
                or self.evaluation_revision is not None
                or self.terminal_status is not None
                or len(artifacts) != 2
                or sum(
                    item.relative_path.endswith("/input.json")
                    for item in artifacts
                )
                != 1
                or self.materialization_manifest is None
                or self.materialization_manifest not in artifacts
                or self.materialization_manifest_digest
                != self.materialization_manifest.sha256
                or self.materialization_id is None
                or self.eval_input_digest is None
                or self.review_target_digest is None
                or self.prepared_source_id is None
                or not visible_files
                or self.adapter_capabilities_digest is None
                or self.target_access is None
                or self.target_access.target_materialization_id
                != self.materialization_id
            ):
                raise SchemaError(
                    "prepare receipt must bind EvalInput and TrialMaterializationManifest"
                )
            expected_suffix = (
                "/materializations/attempt-%04d/materialization_manifest.json"
                % self.attempt
            )
            if not self.materialization_manifest.relative_path.endswith(
                expected_suffix
            ):
                raise SchemaError(
                    "prepare receipt materialization path has the wrong attempt"
                )
        elif self.stage is StageName.AGENT:
            submission_count = sum(
                item.relative_path.endswith("/submission.json")
                for item in artifacts
            )
            if (
                self.attempt is None
                or self.evaluation_id is not None
                or self.evaluation_revision is not None
                or self.terminal_status not in _TERMINAL_STATUSES
                or submission_count != 1
            ):
                raise SchemaError("agent receipt must uniquely commit a terminal Submission")
        elif self.stage is StageName.EVALUATOR:
            if (
                self.attempt is not None
                or self.evaluation_id is None
                or self.evaluation_revision is None
                or self.terminal_status is not None
                or not artifacts
            ):
                raise SchemaError("evaluator receipt has an invalid field combination")
            validate_evaluation_id(
                self.evaluation_id,
                self.run_id,
                self.config_digest,
                self.evaluation_revision,
            )
        if self.stage is not StageName.PREPARE and (
            self.materialization_manifest is not None
            or self.materialization_manifest_digest is not None
            or self.materialization_id is not None
            or self.eval_input_digest is not None
            or self.review_target_digest is not None
            or self.prepared_source_id is not None
            or visible_files
            or self.adapter_capabilities_digest is not None
            or self.target_access is not None
        ):
            raise SchemaError(
                "non-prepare receipt may not carry materialization bindings"
            )
        object.__setattr__(
            self, "artifacts", tuple(sorted(artifacts, key=lambda item: item.relative_path))
        )
        object.__setattr__(
            self,
            "agent_visible_files",
            tuple(sorted(visible_files, key=lambda item: item.relative_path)),
        )
        validate_safe_json(self.to_dict(), "stage_receipt")
        _check_model_size(self, MAX_STAGE_RECEIPT_BYTES, "StageReceipt")

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        task_id: str,
        trial_id: str,
        stage: StageName,
        config_digest: str,
        artifacts: Iterable[ArtifactRef] = (),
        attempt: Optional[int] = None,
        evaluation_id: Optional[str] = None,
        evaluation_revision: Optional[str] = None,
        materialization_manifest: Optional[ArtifactRef] = None,
        materialization_manifest_digest: Optional[str] = None,
        materialization_id: Optional[str] = None,
        eval_input_digest: Optional[str] = None,
        review_target_digest: Optional[str] = None,
        prepared_source_id: Optional[str] = None,
        agent_visible_files: Iterable[AgentVisibleFileBinding] = (),
        adapter_capabilities_digest: Optional[str] = None,
        target_access: Optional[TargetAccess] = None,
        terminal_status: Optional[TrialStatus] = None,
        failure_code: Optional[FailureCode] = None,
    ) -> "StageReceipt":
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            receipt_id=derive_receipt_id(
                run_id,
                task_id,
                trial_id,
                stage,
                config_digest,
                attempt=attempt,
                evaluation_id=evaluation_id,
                evaluation_revision=evaluation_revision,
            ),
            run_id=run_id,
            task_id=task_id,
            trial_id=trial_id,
            stage=stage,
            config_digest=config_digest,
            attempt=attempt,
            evaluation_id=evaluation_id,
            evaluation_revision=evaluation_revision,
            artifacts=_bounded_tuple(
                artifacts,
                "receipt.artifacts",
                MAX_RECEIPT_ARTIFACTS,
            ),
            materialization_manifest=materialization_manifest,
            materialization_manifest_digest=materialization_manifest_digest,
            materialization_id=materialization_id,
            eval_input_digest=eval_input_digest,
            review_target_digest=review_target_digest,
            prepared_source_id=prepared_source_id,
            agent_visible_files=_bounded_tuple(
                agent_visible_files,
                "receipt.agent_visible_files",
                MAX_AGENT_VISIBLE_FILES,
            ),
            adapter_capabilities_digest=adapter_capabilities_digest,
            target_access=target_access,
            terminal_status=terminal_status,
            failure_code=failure_code,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "StageReceipt":
        payload = _object(value, "StageReceipt")
        if "schema_version" in payload:
            _require_protocol_version(
                payload["schema_version"],
                cls.SCHEMA_VERSION,
                "StageReceipt.schema_version",
            )
        _exact_fields(
            payload,
            (
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
                "materialization_manifest",
                "materialization_manifest_digest",
                "materialization_id",
                "eval_input_digest",
                "review_target_digest",
                "prepared_source_id",
                "agent_visible_files",
                "adapter_capabilities_digest",
                "target_access",
                "terminal_status",
                "failure_code",
            ),
            "StageReceipt",
        )
        artifacts = _array(
            payload["artifacts"], "receipt.artifacts", MAX_RECEIPT_ARTIFACTS
        )
        visible_files = _array(
            payload["agent_visible_files"],
            "receipt.agent_visible_files",
            MAX_AGENT_VISIBLE_FILES,
        )
        return cls(
            schema_version=payload["schema_version"],
            receipt_id=validate_path_segment(
                payload["receipt_id"], "receipt.receipt_id"
            ),
            run_id=validate_run_id(payload["run_id"]),
            task_id=_identifier(payload["task_id"], "receipt.task_id"),
            trial_id=validate_trial_id_shape(payload["trial_id"]),
            stage=_enum_value(StageName, payload["stage"], "receipt.stage"),
            config_digest=_digest(payload["config_digest"], "receipt.config_digest"),
            attempt=(
                None
                if payload["attempt"] is None
                else _integer(
                    payload["attempt"],
                    "receipt.attempt",
                    minimum=1,
                    maximum=MAX_TRIAL_ATTEMPTS,
                )
            ),
            evaluation_id=(
                None
                if payload["evaluation_id"] is None
                else validate_evaluation_id_shape(payload["evaluation_id"])
            ),
            evaluation_revision=(
                None
                if payload["evaluation_revision"] is None
                else validate_path_segment(
                    payload["evaluation_revision"],
                    "receipt.evaluation_revision",
                )
            ),
            artifacts=tuple(ArtifactRef.from_dict(item) for item in artifacts),
            materialization_manifest=(
                None
                if payload["materialization_manifest"] is None
                else ArtifactRef.from_dict(payload["materialization_manifest"])
            ),
            materialization_manifest_digest=(
                None
                if payload["materialization_manifest_digest"] is None
                else _digest(
                    payload["materialization_manifest_digest"],
                    "receipt.materialization_manifest_digest",
                )
            ),
            materialization_id=(
                None
                if payload["materialization_id"] is None
                else _identifier(
                    payload["materialization_id"],
                    "receipt.materialization_id",
                )
            ),
            eval_input_digest=(
                None
                if payload["eval_input_digest"] is None
                else _digest(
                    payload["eval_input_digest"],
                    "receipt.eval_input_digest",
                )
            ),
            review_target_digest=(
                None
                if payload["review_target_digest"] is None
                else _digest(
                    payload["review_target_digest"],
                    "receipt.review_target_digest",
                )
            ),
            prepared_source_id=(
                None
                if payload["prepared_source_id"] is None
                else _identifier(
                    payload["prepared_source_id"],
                    "receipt.prepared_source_id",
                )
            ),
            agent_visible_files=tuple(
                AgentVisibleFileBinding.from_dict(item)
                for item in visible_files
            ),
            adapter_capabilities_digest=(
                None
                if payload["adapter_capabilities_digest"] is None
                else _digest(
                    payload["adapter_capabilities_digest"],
                    "receipt.adapter_capabilities_digest",
                )
            ),
            target_access=(
                None
                if payload["target_access"] is None
                else TargetAccess.from_dict(payload["target_access"])
            ),
            terminal_status=(
                None
                if payload["terminal_status"] is None
                else _enum_value(
                    TrialStatus,
                    payload["terminal_status"],
                    "receipt.terminal_status",
                )
            ),
            failure_code=(
                None
                if payload["failure_code"] is None
                else _enum_value(
                    FailureCode, payload["failure_code"], "receipt.failure_code"
                )
            ),
        )

    @classmethod
    def from_json(cls, data: Any) -> "StageReceipt":
        return cls.from_dict(
            _strict_json_loads(data, MAX_STAGE_RECEIPT_BYTES, "StageReceipt JSON")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "trial_id": self.trial_id,
            "stage": self.stage.value,
            "config_digest": self.config_digest,
            "attempt": self.attempt,
            "evaluation_id": self.evaluation_id,
            "evaluation_revision": self.evaluation_revision,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "materialization_manifest": (
                None
                if self.materialization_manifest is None
                else self.materialization_manifest.to_dict()
            ),
            "materialization_manifest_digest": (
                self.materialization_manifest_digest
            ),
            "materialization_id": self.materialization_id,
            "eval_input_digest": self.eval_input_digest,
            "review_target_digest": self.review_target_digest,
            "prepared_source_id": self.prepared_source_id,
            "agent_visible_files": [
                item.to_dict() for item in self.agent_visible_files
            ],
            "adapter_capabilities_digest": (
                self.adapter_capabilities_digest
            ),
            "target_access": (
                None
                if self.target_access is None
                else self.target_access.to_dict()
            ),
            "terminal_status": (
                None if self.terminal_status is None else self.terminal_status.value
            ),
            "failure_code": None if self.failure_code is None else self.failure_code.value,
        }


@dataclass(frozen=True)
class TrialState:
    trial_id: str
    status: TrialStatus
    active_attempt: Optional[int]
    next_attempt: int
    completed_stages: Tuple[StageName, ...]
    terminal_receipt: Optional[StageReceipt]


@dataclass(frozen=True)
class VerifiedTrialMaterialization:
    """Receipt-bound inputs for replaying one active Trial materialization.

    This is the public trust boundary for evaluator composition roots.  The
    path-bearing ArtifactStore internals stay private; consumers receive only
    strictly hydrated canonical values whose Run, Trial, active attempt, and
    committed PREPARE projections have already been cross-checked.
    """

    eval_input: EvalInput
    manifest: TrialMaterializationManifest
    trial_manifest: TrialManifest
    prepare_receipt: StageReceipt
    active_attempt: int
    suite_preparation_binding: Optional[PublicSuitePreparationBindingV2]

    def __post_init__(self) -> None:
        if type(self.eval_input) is not EvalInput:
            raise TypeError("verified materialization requires EvalInput")
        if type(self.manifest) is not TrialMaterializationManifest:
            raise TypeError(
                "verified materialization requires TrialMaterializationManifest"
            )
        if type(self.trial_manifest) is not TrialManifest:
            raise TypeError("verified materialization requires TrialManifest")
        if type(self.prepare_receipt) is not StageReceipt:
            raise TypeError("verified materialization requires StageReceipt")
        if (
            self.suite_preparation_binding is not None
            and type(self.suite_preparation_binding)
            is not PublicSuitePreparationBindingV2
        ):
            raise TypeError("verified materialization preparation binding is invalid")
        if (
            self.active_attempt != self.manifest.attempt
            or self.prepare_receipt.stage is not StageName.PREPARE
            or self.prepare_receipt.attempt != self.active_attempt
            or self.prepare_receipt.materialization_id
            != self.manifest.materialization_id
            or self.eval_input.digest() != self.manifest.eval_input_digest
            or self.trial_manifest.eval_input_digest
            != self.manifest.eval_input_digest
        ):
            raise ArtifactIntegrityError(
                "verified Trial materialization binding is inconsistent"
            )

@dataclass(frozen=True)
class RunState:
    run_id: str
    status: RunStatus
    trials: Tuple[TrialState, ...]


@dataclass(frozen=True)
class ResumePlan:
    trial_id: str
    status: TrialStatus
    completed_stages: Tuple[StageName, ...]
    missing_stages: Tuple[StageName, ...]
    terminal: bool


def _canonical_payload_text(
    value: Any,
    context: str,
    *,
    evaluator_context_policy: Optional[str] = None,
) -> str:
    validate_safe_json(
        value,
        context,
        evaluator_context_policy=_evaluator_context_policy_for_payload(
            value,
            evaluator_context_policy,
        ),
    )
    return canonical_json_bytes(value).decode("utf-8", "strict")


def _validated_artifact_text(
    value: Any,
    context: str,
    *,
    allow_rendered_environment_projection: bool = False,
) -> str:
    """Apply the secret boundary while tolerating canonical metric ``NAME=`` text.

    The Run report renderer can legitimately emit several uppercase metric or
    policy labels followed by ``=`` inside canonical JSON projections.  That
    resembles a full environment dump to the generic heuristic.  Suppression
    is narrowly limited to that final heuristic: URL userinfo, credentials,
    secret values, and raw reasoning are checked first and still fail.
    """

    try:
        return validate_safe_text(value, context)
    except SchemaError as exc:
        if (
            allow_rendered_environment_projection
            and str(exc)
            == "%s contains a forbidden full environment dump" % context
            and type(value) is str
        ):
            return value
        raise


def _decoded_payload(value: str) -> Any:
    # Values are stored only after canonical UTF-8 validation.  Returning a
    # fresh tree prevents callers from mutating the frozen bundle snapshot.
    return json.loads(value)


@dataclass(frozen=True)
class EvaluationNamespace:
    """Path-free metadata for one committed Trial evaluation namespace."""

    run_id: str
    task_id: str
    trial_id: str
    evaluation_id: str
    evaluation_revision: str
    evaluator_execution_digest: str
    receipt: StageReceipt

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        _identifier(self.task_id, "evaluation namespace.task_id")
        validate_trial_id_shape(self.trial_id)
        validate_evaluation_id(
            self.evaluation_id,
            self.run_id,
            self.evaluator_execution_digest,
            self.evaluation_revision,
        )
        if type(self.receipt) is not StageReceipt:
            raise SchemaError(
                "evaluation namespace receipt must be a StageReceipt"
            )
        if (
            self.receipt.stage is not StageName.EVALUATOR
            or self.receipt.run_id != self.run_id
            or self.receipt.task_id != self.task_id
            or self.receipt.trial_id != self.trial_id
            or self.receipt.evaluation_id != self.evaluation_id
            or self.receipt.evaluation_revision != self.evaluation_revision
            or self.receipt.config_digest != self.evaluator_execution_digest
        ):
            raise SchemaError(
                "evaluation namespace receipt binding is inconsistent"
            )
        names = []
        prefix = "cases/%s/trials/%s/evaluations/%s/" % (
            derive_case_path_id(self.task_id),
            self.trial_id,
            self.evaluation_id,
        )
        for artifact in self.receipt.artifacts:
            filename = artifact.relative_path.rsplit("/", 1)[-1]
            if artifact.relative_path != prefix + filename:
                raise SchemaError(
                    "evaluation namespace artifact has the wrong path binding"
                )
            names.append(filename)
        required = set(_EVALUATION_JSON_ARTIFACT_NAMES)
        actual = set(names)
        if (
            len(names) != len(actual)
            or not required.issubset(actual)
            or not actual.issubset(
                required | set(_EVALUATION_OPTIONAL_ARTIFACT_NAMES)
            )
        ):
            raise SchemaError(
                "evaluation namespace has an invalid artifact set"
            )

    @property
    def artifacts(self) -> Tuple[ArtifactRef, ...]:
        return self.receipt.artifacts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "trial_id": self.trial_id,
            "evaluation_id": self.evaluation_id,
            "evaluation_revision": self.evaluation_revision,
            "evaluator_execution_digest": self.evaluator_execution_digest,
            "receipt": self.receipt.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationArtifactBundle:
    """Canonical, source-bound contents of one committed Trial evaluation."""

    namespace: EvaluationNamespace
    evaluator_execution: EvaluatorExecutionConfig
    submission_digest: str
    canonical_case_digest: str
    trial_manifest_digest: str
    _intent_matches_json: str = field(repr=False)
    _review_matches_json: str = field(repr=False)
    _judge_input_json: str = field(repr=False)
    _judge_output_json: str = field(repr=False)
    _score_json: str = field(repr=False)
    _report: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.namespace) is not EvaluationNamespace:
            raise TypeError("evaluation bundle namespace is invalid")
        if type(self.evaluator_execution) is not EvaluatorExecutionConfig:
            raise TypeError("evaluation bundle execution config is invalid")
        if (
            self.evaluator_execution.digest()
            != self.namespace.evaluator_execution_digest
        ):
            raise ArtifactIntegrityError(
                "evaluation execution config differs from namespace receipt"
            )
        _digest(self.submission_digest, "evaluation bundle.submission_digest")
        _digest(
            self.canonical_case_digest,
            "evaluation bundle.canonical_case_digest",
        )
        _digest(
            self.trial_manifest_digest,
            "evaluation bundle.trial_manifest_digest",
        )
        for name in (
            "_intent_matches_json",
            "_review_matches_json",
            "_judge_input_json",
            "_judge_output_json",
            "_score_json",
        ):
            value = getattr(self, name)
            if type(value) is not str:
                raise TypeError("evaluation bundle payload snapshots must be strings")
            decoded = _decoded_payload(value)
            if _canonical_payload_text(
                decoded,
                "evaluation bundle payload",
                evaluator_context_policy=(
                    _EVALUATOR_CONTEXT_POLICY_BY_BUNDLE_FIELD.get(name)
                ),
            ) != value:
                raise ArtifactIntegrityError(
                    "evaluation bundle payload is not canonical"
                )
        if self._report is not None:
            report = validate_safe_text(self._report, "evaluation report")
            if "\r" in report:
                raise ArtifactIntegrityError(
                    "evaluation report must use canonical LF line endings"
                )

    @property
    def run_id(self) -> str:
        return self.namespace.run_id

    @property
    def task_id(self) -> str:
        return self.namespace.task_id

    @property
    def trial_id(self) -> str:
        return self.namespace.trial_id

    @property
    def evaluation_id(self) -> str:
        return self.namespace.evaluation_id

    @property
    def evaluation_revision(self) -> str:
        return self.namespace.evaluation_revision

    @property
    def receipt(self) -> StageReceipt:
        return self.namespace.receipt

    @property
    def intent_matches(self) -> Any:
        return _decoded_payload(self._intent_matches_json)

    @property
    def review_matches(self) -> Any:
        return _decoded_payload(self._review_matches_json)

    @property
    def judge_input(self) -> Any:
        return _decoded_payload(self._judge_input_json)

    @property
    def judge_output(self) -> Any:
        return _decoded_payload(self._judge_output_json)

    @property
    def score(self) -> Any:
        return _decoded_payload(self._score_json)

    @property
    def report(self) -> Optional[str]:
        return self._report

    def to_dict(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace.to_dict(),
            "evaluator_execution": self.evaluator_execution.to_dict(),
            "submission_digest": self.submission_digest,
            "canonical_case_digest": self.canonical_case_digest,
            "trial_manifest_digest": self.trial_manifest_digest,
            "intent_matches": self.intent_matches,
            "review_matches": self.review_matches,
            "judge_input": self.judge_input,
            "judge_output": self.judge_output,
            "score": self.score,
            "report": self.report,
        }


@dataclass(frozen=True)
class RunEvaluationNamespace:
    """Metadata for one committed Run-level summary/report pair."""

    schema_version: str
    run_id: str
    evaluation_id: str
    evaluation_revision: str
    evaluator_execution_digest: str
    summary_id: str
    summary: ArtifactRef
    report: ArtifactRef

    def __post_init__(self) -> None:
        _require_protocol_version(
            self.schema_version,
            EVAL_RUN_EVALUATION_NAMESPACE_SCHEMA_VERSION,
            "RunEvaluationNamespace.schema_version",
        )
        validate_run_id(self.run_id)
        validate_evaluation_id(
            self.evaluation_id,
            self.run_id,
            self.evaluator_execution_digest,
            self.evaluation_revision,
        )
        validate_path_segment(self.summary_id, "Run evaluation summary_id")
        if type(self.summary) is not ArtifactRef or type(self.report) is not ArtifactRef:
            raise SchemaError("Run evaluation namespace refs are invalid")
        prefix = "evaluations/%s/" % self.evaluation_id
        if (
            self.summary.relative_path != prefix + "summary.json"
            or self.report.relative_path != prefix + "report.md"
        ):
            raise SchemaError("Run evaluation namespace refs have wrong paths")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "evaluation_id": self.evaluation_id,
            "evaluation_revision": self.evaluation_revision,
            "evaluator_execution_digest": self.evaluator_execution_digest,
            "summary_id": self.summary_id,
            "summary": self.summary.to_dict(),
            "report": self.report.to_dict(),
        }


@dataclass(frozen=True)
class RunEvaluationBundle:
    """Canonical Run-level report projection without filesystem path handles."""

    namespace: RunEvaluationNamespace
    _summary_json: str = field(repr=False)
    _report: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.namespace) is not RunEvaluationNamespace:
            raise TypeError("Run evaluation bundle namespace is invalid")
        summary = _decoded_payload(self._summary_json)
        if type(summary) is not dict:
            raise ArtifactIntegrityError(
                "Run evaluation summary snapshot is not an object"
            )
        if _canonical_payload_text(summary, "Run evaluation summary") != self._summary_json:
            raise ArtifactIntegrityError(
                "Run evaluation summary snapshot is not canonical"
            )
        if summary.get("summary_id") != self.namespace.summary_id:
            raise ArtifactIntegrityError(
                "Run evaluation summary ID differs from namespace"
            )
        report = _validated_artifact_text(
            self._report,
            "Run evaluation report",
            allow_rendered_environment_projection=True,
        )
        if "\r" in report:
            raise ArtifactIntegrityError(
                "Run evaluation report must use canonical LF line endings"
            )

    @property
    def summary(self) -> Dict[str, Any]:
        value = _decoded_payload(self._summary_json)
        if type(value) is not dict:
            raise ArtifactIntegrityError("Run evaluation summary is not an object")
        return value

    @property
    def report(self) -> str:
        return self._report

    def hydrate_summary(self, **sources: Any) -> Any:
        """Replay strict report hydration against caller-supplied root sources."""

        from .report import RunReportSummary, render_run_markdown

        hydrated = RunReportSummary.from_dict(self.summary, **sources)
        if render_run_markdown(hydrated) != self.report:
            raise ArtifactIntegrityError(
                "persisted Run report differs from source-bound summary rendering"
            )
        return hydrated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace.to_dict(),
            "summary": self.summary,
            "report": self.report,
        }


@dataclass(frozen=True)
class _VerifiedRunBundle:
    manifest: RunManifest
    config: EvalRunConfig
    case_snapshot: RunCaseSnapshot


@dataclass
class _ReadBudget:
    maximum: int
    consumed: int = 0

    def ensure(self, amount: int) -> None:
        if amount < 0 or self.consumed + amount > self.maximum:
            raise ArtifactIntegrityError("artifact reads exceed the cumulative byte limit")

    def add(self, amount: int) -> None:
        self.ensure(amount)
        self.consumed += amount


def _is_reparse(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)


def _unsafe_node(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or _is_reparse(info)


def _file_identity(info: os.stat_result) -> Optional[Tuple[int, int]]:
    inode = getattr(info, "st_ino", 0)
    if not inode:
        return None
    return (getattr(info, "st_dev", 0), inode)


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare usable filesystem identities; missing inode data is unknown."""

    left_identity = _file_identity(left)
    right_identity = _file_identity(right)
    return (
        left_identity is not None
        and right_identity is not None
        and left_identity == right_identity
    )


def _normalized_filesystem_path(value: os.PathLike[str] | str) -> str:
    path = os.path.abspath(os.fspath(value))
    if os.name == "nt":
        if path.startswith("\\\\?\\UNC\\"):
            path = "\\\\" + path[8:]
        elif path.startswith("\\\\?\\"):
            path = path[4:]
    return os.path.normcase(os.path.normpath(path))


def _path_is_within(root: Path, target: Path) -> bool:
    root_text = _normalized_filesystem_path(root)
    target_text = _normalized_filesystem_path(target)
    try:
        common = os.path.commonpath((root_text, target_text))
    except ValueError:
        return False
    return os.path.normcase(common) == root_text


def _windows_descriptor_path(descriptor: int) -> Optional[Path]:
    """Return the kernel-resolved path for an opened Windows descriptor."""

    if os.name != "nt":
        return None
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_final_path = kernel32.GetFinalPathNameByHandleW
        get_final_path.argtypes = (
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        get_final_path.restype = wintypes.DWORD
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
        required = get_final_path(handle, None, 0, 0)
        if not required:
            raise OSError(
                ctypes.get_last_error(), "GetFinalPathNameByHandleW failed"
            )
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = get_final_path(handle, buffer, len(buffer), 0)
        if not written or written >= len(buffer):
            raise OSError(
                ctypes.get_last_error(), "GetFinalPathNameByHandleW failed"
            )
        value = buffer.value
    except (ImportError, OSError, ValueError) as exc:
        raise ArtifactSecurityError(
            "could not verify the opened Windows artifact identity"
        ) from exc
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


def _windows_descriptor_identity(descriptor: int) -> Optional[Tuple[int, int]]:
    """Return the stable Windows volume/file ID for an open descriptor."""

    if os.name != "nt":
        return None
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class ByHandleFileInformation(ctypes.Structure):
            _fields_ = (
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
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_information = kernel32.GetFileInformationByHandle
        get_information.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ByHandleFileInformation),
        )
        get_information.restype = wintypes.BOOL
        information = ByHandleFileInformation()
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
        if not get_information(handle, ctypes.byref(information)):
            raise OSError(
                ctypes.get_last_error(), "GetFileInformationByHandle failed"
            )
        file_index = (
            int(information.nFileIndexHigh) << 32
        ) | int(information.nFileIndexLow)
        if file_index == 0:
            raise OSError("Windows file identity is unavailable")
        return (int(information.dwVolumeSerialNumber), file_index)
    except (ImportError, OSError, ValueError) as exc:
        raise ArtifactSecurityError(
            "could not verify the opened Windows artifact file ID"
        ) from exc


def _descriptor_identity(
    descriptor: int, metadata: os.stat_result
) -> Optional[Tuple[int, int]]:
    identity = _file_identity(metadata)
    if identity is not None:
        return identity
    return _windows_descriptor_identity(descriptor)


def _windows_open_directory_handle(path: Path) -> int:
    if os.name != "nt":
        raise ArtifactSecurityError("Windows directory handles are unavailable")
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            str(path),
            0x0080,  # FILE_READ_ATTRIBUTES
            0x00000001 | 0x00000002,  # share read/write, deliberately not delete
            None,
            3,  # OPEN_EXISTING
            0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        handle_value = (
            int(handle)
            if isinstance(handle, int)
            else int(getattr(handle, "value", 0) or 0)
        )
        if not handle_value or handle_value == invalid:
            raise OSError(ctypes.get_last_error(), "CreateFileW directory failed")
        return handle_value
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise ArtifactSecurityError(
            "could not acquire a non-replaceable Windows directory handle"
        ) from exc


def _windows_close_handle(handle: int) -> None:
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    close_handle(wintypes.HANDLE(handle))


def _windows_raw_handle_path(handle: int) -> Path:
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_final_path = kernel32.GetFinalPathNameByHandleW
        get_final_path.argtypes = (
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        get_final_path.restype = wintypes.DWORD
        raw_handle = wintypes.HANDLE(handle)
        required = get_final_path(raw_handle, None, 0, 0)
        if not required:
            raise OSError(ctypes.get_last_error(), "GetFinalPathNameByHandleW failed")
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = get_final_path(raw_handle, buffer, len(buffer), 0)
        if not written or written >= len(buffer):
            raise OSError(ctypes.get_last_error(), "GetFinalPathNameByHandleW failed")
        value = buffer.value
    except (ImportError, OSError, ValueError) as exc:
        raise ArtifactSecurityError(
            "could not verify a Windows directory handle path"
        ) from exc
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


def _windows_raw_handle_attributes(handle: int) -> int:
    try:
        import ctypes
        from ctypes import wintypes

        class ByHandleFileInformation(ctypes.Structure):
            _fields_ = (
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
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_information = kernel32.GetFileInformationByHandle
        get_information.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ByHandleFileInformation),
        )
        get_information.restype = wintypes.BOOL
        information = ByHandleFileInformation()
        if not get_information(
            wintypes.HANDLE(handle), ctypes.byref(information)
        ):
            raise OSError(
                ctypes.get_last_error(), "GetFileInformationByHandle failed"
            )
        return int(information.dwFileAttributes)
    except (ImportError, OSError, ValueError) as exc:
        raise ArtifactSecurityError(
            "could not verify Windows directory handle attributes"
        ) from exc


def _absolute_storage_path(value: os.PathLike[str] | str) -> Path:
    """Return an absolute path, using Win32 extended-length form on Windows."""

    raw = os.fspath(value)
    if os.name != "nt":
        return Path(os.path.abspath(raw))
    if raw.startswith("\\\\?\\"):
        return Path(raw)
    absolute = os.path.abspath(raw)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


class ArtifactStore:
    """Fail-closed store rooted at an explicit ``.eval-runs`` directory.

    File contents are flushed before create-only publication on every platform.
    Parent-directory metadata is additionally fsynced where the Python runtime
    exposes POSIX directory descriptors.  Windows therefore provides atomic
    no-overwrite publication and file flush, but this class does not claim the
    stronger POSIX parent-directory durability guarantee there.
    """

    def __init__(
        self,
        runs_root: os.PathLike[str] | str,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_total_read_bytes: int = DEFAULT_MAX_TOTAL_READ_BYTES,
        create_root: bool = True,
    ) -> None:
        if type(max_file_bytes) is not int or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        if type(max_total_read_bytes) is not int or max_total_read_bytes <= 0:
            raise ValueError("max_total_read_bytes must be a positive integer")
        if max_file_bytes > max_total_read_bytes:
            raise ValueError("max_file_bytes may not exceed max_total_read_bytes")
        if type(create_root) is not bool:
            raise ValueError("create_root must be a bool")
        root = _absolute_storage_path(runs_root)
        if root.name != ".eval-runs":
            raise ValueError("ArtifactStore root must be an explicit .eval-runs directory")
        self.root = root
        self.max_file_bytes = max_file_bytes
        self.max_total_read_bytes = max_total_read_bytes
        if create_root:
            self._prepare_root()
        else:
            self._require_existing_root()
        root_metadata = os.lstat(self.root)
        self._root_identity = _file_identity(root_metadata)
        if os.name != "nt" and self._root_identity is None:
            raise ArtifactSecurityError(
                "artifact root filesystem identity is unavailable"
            )

    @property
    def directory_fsync_supported(self) -> bool:
        return DIRECTORY_FSYNC_SUPPORTED

    def _prepare_root(self) -> None:
        current = self.root.parent
        while True:
            try:
                ancestor = os.lstat(current)
            except OSError as exc:
                raise ArtifactSecurityError("artifact root ancestor is unavailable") from exc
            if _unsafe_node(ancestor) or not stat.S_ISDIR(ancestor.st_mode):
                raise ArtifactSecurityError(
                    "artifact root ancestor is a link, reparse point, or non-directory"
                )
            parent = current.parent
            if parent == current:
                break
            current = parent
        created = False
        try:
            os.mkdir(self.root, 0o700)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise ArtifactSecurityError("could not create artifact root") from exc
        self._assert_directory(self.root)
        if created and DIRECTORY_FSYNC_SUPPORTED:
            self._fsync_directory(self.root.parent)

    def _require_existing_root(self) -> None:
        try:
            metadata = os.lstat(self.root)
        except FileNotFoundError as exc:
            raise ArtifactIntegrityError(
                "read-only artifact root does not exist"
            ) from exc
        except OSError as exc:
            raise ArtifactSecurityError(
                "read-only artifact root is unavailable"
            ) from exc
        if _unsafe_node(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactSecurityError(
                "read-only artifact root is a link, reparse point, or non-directory"
            )

    @staticmethod
    def _validate_internal_component(value: str) -> None:
        if value in _INTERNAL_DIRECTORIES or _ATTEMPT_RE.fullmatch(value):
            return
        validate_path_segment(value, "artifact directory")

    def _within_root(self, path: Path) -> Path:
        absolute = _absolute_storage_path(path)
        try:
            common = os.path.commonpath((os.fspath(self.root), os.fspath(absolute)))
        except ValueError as exc:
            raise ArtifactSecurityError("artifact path crosses filesystem roots") from exc
        if os.path.normcase(common) != os.path.normcase(os.fspath(self.root)):
            raise ArtifactSecurityError("artifact path escapes .eval-runs")
        return absolute

    def _assert_directory(self, path: Path) -> None:
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise ArtifactSecurityError("required artifact directory is unavailable") from exc
        if _unsafe_node(info) or not stat.S_ISDIR(info.st_mode):
            raise ArtifactSecurityError(
                "artifact path contains a symlink, reparse point, or non-directory"
            )

    def _ensure_directory(self, path: Path) -> None:
        path = self._within_root(path)
        if os.name == "nt":
            handles: List[int] = []
            current = self.root
            try:
                root_handle = _windows_open_directory_handle(current)
                handles.append(root_handle)
                self._validate_windows_directory_handle(root_handle, current)
                for part in path.relative_to(self.root).parts:
                    self._validate_internal_component(part)
                    target = current / part
                    try:
                        os.mkdir(target, 0o700)
                    except FileExistsError:
                        pass
                    except OSError as exc:
                        raise ArtifactSecurityError(
                            "could not create artifact directory"
                        ) from exc
                    handle = _windows_open_directory_handle(target)
                    handles.append(handle)
                    self._validate_windows_directory_handle(handle, target)
                    current = target
            finally:
                for handle in reversed(handles):
                    _windows_close_handle(handle)
            return

        if os.open not in getattr(os, "supports_dir_fd", set()):
            raise ArtifactSecurityError(
                "descriptor-relative directory creation is unavailable"
            )
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(self.root, directory_flags)
        try:
            if _file_identity(os.fstat(descriptor)) != self._root_identity:
                raise ArtifactSecurityError(
                    "artifact root identity changed after initialization"
                )
            for part in path.relative_to(self.root).parts:
                self._validate_internal_component(part)
                created = False
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise ArtifactSecurityError(
                        "could not create artifact directory"
                    ) from exc
                if created:
                    os.fsync(descriptor)
                next_descriptor = os.open(
                    part, directory_flags, dir_fd=descriptor
                )
                metadata = os.fstat(next_descriptor)
                if _unsafe_node(metadata) or not stat.S_ISDIR(metadata.st_mode):
                    os.close(next_descriptor)
                    raise ArtifactSecurityError(
                        "artifact path contains an unsafe directory component"
                    )
                os.close(descriptor)
                descriptor = next_descriptor
        finally:
            os.close(descriptor)

    def _validate_windows_directory_handle(
        self, handle: int, expected_path: Path
    ) -> None:
        actual = _windows_raw_handle_path(handle)
        attributes = _windows_raw_handle_attributes(handle)
        if (
            _normalized_filesystem_path(actual)
            != _normalized_filesystem_path(expected_path)
            or not _path_is_within(self.root, actual)
        ):
            raise ArtifactSecurityError(
                "Windows artifact directory resolved to an unexpected path"
            )
        if attributes & _REPARSE_POINT or not attributes & 0x10:
            raise ArtifactSecurityError(
                "Windows artifact directory is a reparse point or non-directory"
            )

    def _open_posix_directory_descriptor(self, path: Path) -> int:
        if os.name == "nt" or os.open not in getattr(os, "supports_dir_fd", set()):
            raise ArtifactSecurityError(
                "descriptor-relative directory access is unavailable"
            )
        path = self._within_root(path)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(self.root, flags)
        try:
            if _file_identity(os.fstat(descriptor)) != self._root_identity:
                raise ArtifactSecurityError(
                    "artifact root identity changed after initialization"
                )
            for component in path.relative_to(self.root).parts:
                next_descriptor = os.open(
                    component, flags, dir_fd=descriptor
                )
                metadata = os.fstat(next_descriptor)
                if _unsafe_node(metadata) or not stat.S_ISDIR(metadata.st_mode):
                    os.close(next_descriptor)
                    raise ArtifactSecurityError(
                        "artifact path contains an unsafe directory component"
                    )
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @contextmanager
    def _guard_parent_directory(self, path: Path) -> Iterator[Optional[int]]:
        path = self._within_root(path)
        parent = path.parent
        self._ensure_directory(parent)
        if os.name != "nt":
            descriptor = self._open_posix_directory_descriptor(parent)
            try:
                yield descriptor
            finally:
                os.close(descriptor)
            return

        handles: List[int] = []
        current = self.root
        try:
            root_handle = _windows_open_directory_handle(current)
            handles.append(root_handle)
            self._validate_windows_directory_handle(root_handle, current)
            for component in parent.relative_to(self.root).parts:
                current = current / component
                handle = _windows_open_directory_handle(current)
                handles.append(handle)
                self._validate_windows_directory_handle(handle, current)
            yield None
        finally:
            for handle in reversed(handles):
                _windows_close_handle(handle)

    def _assert_parent_chain(self, path: Path) -> None:
        path = self._within_root(path)
        current = self.root
        self._assert_directory(current)
        for part in path.parent.relative_to(self.root).parts:
            current = current / part
            self._assert_directory(current)

    def _exists_regular(self, path: Path) -> bool:
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise ArtifactSecurityError("artifact path could not be inspected") from exc
        self._assert_parent_chain(path)
        if _unsafe_node(info) or not stat.S_ISREG(info.st_mode):
            raise ArtifactSecurityError(
                "artifact is a symlink, reparse point, or special file"
            )
        return True

    def _run_dir(self, run_id: str) -> Path:
        validate_run_id(run_id)
        return self.root / run_id

    def _trial_dir(self, plan: TrialManifest) -> Path:
        return (
            self._run_dir(plan.run_id)
            / "cases"
            / plan.case_path_id
            / "trials"
            / plan.trial_id
        )

    def _target(self, run_id: str, relative_path: str) -> Path:
        relative = _relative_artifact_path(relative_path)
        return self._within_root(self._run_dir(run_id) / Path(*relative.split("/")))

    def _run_relative(self, run_id: str, path: Path) -> str:
        return _relative_artifact_path(
            self._within_root(path).relative_to(self._run_dir(run_id)).as_posix()
        )

    @staticmethod
    def _write_all(descriptor: int, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short artifact write")
            offset += written

    @staticmethod
    def _fsync_directory(path: Path) -> bool:
        """Flush parent metadata when supported and report that capability."""

        if not DIRECTORY_FSYNC_SUPPORTED:
            return False
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return True

    def _write_bytes_exclusive(self, path: Path, data: bytes) -> None:
        """Atomically publish bytes with create-if-absent semantics."""

        if len(data) > self.max_file_bytes:
            raise ArtifactIntegrityError("artifact exceeds the single-file byte limit")
        path = self._within_root(path)
        temp = path.parent / (".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
        temp_name = temp.name
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor: Optional[int] = None
        parent_descriptor: Optional[int] = None
        with self._guard_parent_directory(path) as guarded_parent:
            parent_descriptor = guarded_parent
            try:
                if parent_descriptor is None:
                    descriptor = os.open(temp, flags, 0o600)
                else:
                    descriptor = os.open(
                        temp_name,
                        flags,
                        0o600,
                        dir_fd=parent_descriptor,
                    )
                self._write_all(descriptor, data)
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = None
                try:
                    if parent_descriptor is None:
                        # The guarded Windows parent chain cannot be renamed or
                        # replaced while this no-overwrite rename is in flight.
                        os.rename(temp, path)
                    else:
                        os.link(
                            temp_name,
                            path.name,
                            src_dir_fd=parent_descriptor,
                            dst_dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                        os.unlink(temp_name, dir_fd=parent_descriptor)
                except FileExistsError as exc:
                    raise ArtifactConflictError(
                        "create-only artifact already exists"
                    ) from exc
                except PermissionError as exc:
                    if os.path.lexists(path):
                        raise ArtifactConflictError(
                            "artifact conflicts with an existing writer or completed file"
                        ) from exc
                    raise ArtifactSecurityError(
                        "artifact publication was denied"
                    ) from exc
                if parent_descriptor is None:
                    # Do not claim directory fsync on Windows; the capability is
                    # explicitly exposed as false.
                    self._fsync_directory(path.parent)
                else:
                    os.fsync(parent_descriptor)
            except ArtifactError:
                raise
            except OSError as exc:
                if exc.errno in (errno.EEXIST, errno.EACCES) and os.path.lexists(path):
                    raise ArtifactConflictError(
                        "create-only artifact already exists"
                    ) from exc
                raise ArtifactError("atomic create-if-absent write failed") from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                try:
                    if parent_descriptor is None:
                        os.unlink(temp)
                    else:
                        os.unlink(temp_name, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    @contextmanager
    def _lock(self, path: Path) -> Iterator[None]:
        path = self._within_root(path)
        with self._guard_parent_directory(path) as parent_descriptor:
            try:
                if parent_descriptor is None:
                    existing = os.lstat(path)
                else:
                    existing = os.stat(
                        path.name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
            except FileNotFoundError:
                existing = None
            except OSError as exc:
                raise ArtifactSecurityError("could not inspect writer lock") from exc
            if existing is not None and (
                _unsafe_node(existing) or not stat.S_ISREG(existing.st_mode)
            ):
                raise ArtifactSecurityError(
                    "writer lock is a symlink, reparse point, or special file"
                )
            flags = (
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                if parent_descriptor is None:
                    descriptor = os.open(path, flags, 0o600)
                else:
                    descriptor = os.open(
                        path.name,
                        flags,
                        0o600,
                        dir_fd=parent_descriptor,
                    )
            except OSError as exc:
                raise ArtifactSecurityError("could not open writer lock") from exc
            locked = False
            try:
                info = os.fstat(descriptor)
                if _unsafe_node(info) or not stat.S_ISREG(info.st_mode):
                    raise ArtifactSecurityError("writer lock is not a regular file")
                opened_path = _windows_descriptor_path(descriptor)
                if opened_path is not None and (
                    not _path_is_within(self.root, opened_path)
                    or _normalized_filesystem_path(opened_path)
                    != _normalized_filesystem_path(path)
                ):
                    raise ArtifactSecurityError(
                        "writer lock resolved to an unexpected path"
                    )
                if info.st_size == 0:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    os.write(descriptor, b"\0")
                    os.fsync(descriptor)
                os.lseek(descriptor, 0, os.SEEK_SET)
                try:
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except (OSError, BlockingIOError) as exc:
                    raise ArtifactConflictError(
                        "another writer owns this namespace"
                    ) from exc
                yield
            finally:
                if locked:
                    try:
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        if os.name == "nt":
                            import msvcrt

                            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                    except OSError:
                        pass
                os.close(descriptor)

    def _open_read_descriptor(self, path: Path) -> int:
        """Open a file without trusting a previously inspected parent chain."""

        path = self._within_root(path)
        relative = path.relative_to(self.root)
        components = relative.parts
        if not components:
            raise ArtifactSecurityError("artifact path must name a file")
        binary = getattr(os, "O_BINARY", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        cloexec = getattr(os, "O_CLOEXEC", 0)
        if os.name != "nt" and os.open in getattr(os, "supports_dir_fd", set()):
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | nofollow
                | cloexec
            )
            directory_descriptor = os.open(self.root, directory_flags)
            try:
                if (
                    _file_identity(os.fstat(directory_descriptor))
                    != self._root_identity
                ):
                    raise ArtifactSecurityError(
                        "artifact root identity changed after initialization"
                    )
                for component in components[:-1]:
                    next_descriptor = os.open(
                        component,
                        directory_flags,
                        dir_fd=directory_descriptor,
                    )
                    os.close(directory_descriptor)
                    directory_descriptor = next_descriptor
                    metadata = os.fstat(directory_descriptor)
                    if _unsafe_node(metadata) or not stat.S_ISDIR(metadata.st_mode):
                        raise ArtifactSecurityError(
                            "artifact path contains an unsafe directory component"
                        )
                return os.open(
                    components[-1],
                    os.O_RDONLY | binary | nofollow | cloexec,
                    dir_fd=directory_descriptor,
                )
            finally:
                os.close(directory_descriptor)
        return os.open(path, os.O_RDONLY | binary | nofollow | cloexec)

    def _read_bytes(
        self,
        path: Path,
        *,
        expected_sha256: Optional[str],
        expected_size: Optional[int],
        budget: _ReadBudget,
        maximum_bytes: Optional[int] = None,
    ) -> bytes:
        effective_maximum = min(
            self.max_file_bytes,
            self.max_file_bytes if maximum_bytes is None else maximum_bytes,
        )
        path = self._within_root(path)
        self._assert_parent_chain(path)
        try:
            before = os.lstat(path)
        except OSError as exc:
            raise ArtifactIntegrityError("required artifact is missing") from exc
        if _unsafe_node(before) or not stat.S_ISREG(before.st_mode):
            raise ArtifactSecurityError(
                "artifact is a symlink, reparse point, or special file"
            )
        if before.st_size > effective_maximum:
            raise ArtifactIntegrityError("artifact exceeds the single-file byte limit")
        if expected_size is not None and before.st_size != expected_size:
            raise ArtifactIntegrityError("artifact size does not match its descriptor")
        budget.ensure(before.st_size)
        try:
            descriptor = self._open_read_descriptor(path)
        except OSError as exc:
            raise ArtifactSecurityError("artifact could not be safely opened") from exc
        chunks: List[bytes] = []
        total = 0
        path_revalidated_by_handle = False
        try:
            opened = os.fstat(descriptor)
            if _unsafe_node(opened) or not stat.S_ISREG(opened.st_mode):
                raise ArtifactSecurityError("artifact changed during safe open")
            before_identity = _file_identity(before)
            opened_identity = _descriptor_identity(descriptor, opened)
            if (
                before_identity is not None
                and opened_identity is not None
                and before_identity != opened_identity
            ):
                raise ArtifactSecurityError("artifact changed during safe open")
            opened_path = _windows_descriptor_path(descriptor)
            if opened_path is not None:
                if not _path_is_within(self.root, opened_path):
                    raise ArtifactSecurityError(
                        "opened artifact resolved outside .eval-runs"
                    )
                if _normalized_filesystem_path(opened_path) != _normalized_filesystem_path(
                    path
                ):
                    raise ArtifactSecurityError(
                        "opened artifact resolved to an unexpected path"
                    )
            elif before_identity is None or opened_identity is None:
                raise ArtifactSecurityError(
                    "artifact filesystem identity could not be verified"
                )
            if before.st_size != opened.st_size:
                raise ArtifactSecurityError("artifact changed during safe open")
            if opened.st_size > effective_maximum:
                raise ArtifactIntegrityError(
                    "artifact exceeds the single-file byte limit"
                )
            if expected_size is not None and opened.st_size != expected_size:
                raise ArtifactIntegrityError(
                    "artifact size does not match its descriptor"
                )
            budget.ensure(opened.st_size)
            while True:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, max(1, effective_maximum + 1 - total)),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > effective_maximum:
                    raise ArtifactIntegrityError(
                        "artifact exceeds the single-file byte limit"
                    )
                budget.ensure(total)
            after = os.fstat(descriptor)
            after_identity = _descriptor_identity(descriptor, after)
            if (
                opened_identity is not None
                and after_identity is not None
                and opened_identity != after_identity
            ):
                raise ArtifactIntegrityError("artifact changed while being read")
            if opened_identity is None or after_identity is None:
                if opened_path is None:
                    raise ArtifactSecurityError(
                        "artifact filesystem identity became unverifiable"
                    )
            if (
                after.st_size != total
                or opened.st_size != after.st_size
                or getattr(opened, "st_mtime_ns", None)
                != getattr(after, "st_mtime_ns", None)
            ):
                raise ArtifactIntegrityError("artifact changed while being read")
            if opened_path is not None:
                recheck_descriptor: Optional[int] = None
                try:
                    recheck_descriptor = self._open_read_descriptor(path)
                    recheck = os.fstat(recheck_descriptor)
                    if _unsafe_node(recheck) or not stat.S_ISREG(recheck.st_mode):
                        raise ArtifactSecurityError(
                            "artifact path changed while it was open"
                        )
                    recheck_path = _windows_descriptor_path(recheck_descriptor)
                    if (
                        recheck_path is None
                        or _normalized_filesystem_path(recheck_path)
                        != _normalized_filesystem_path(path)
                    ):
                        raise ArtifactSecurityError(
                            "artifact path resolved to an unexpected file during revalidation"
                        )
                    recheck_identity = _descriptor_identity(
                        recheck_descriptor, recheck
                    )
                    if (
                        opened_identity is None
                        or recheck_identity is None
                        or opened_identity != recheck_identity
                    ):
                        raise ArtifactSecurityError(
                            "artifact path changed while it was open"
                        )
                    path_revalidated_by_handle = True
                finally:
                    if recheck_descriptor is not None:
                        os.close(recheck_descriptor)
        finally:
            os.close(descriptor)
        self._assert_parent_chain(path)
        try:
            path_after = os.lstat(path)
        except OSError as exc:
            raise ArtifactSecurityError(
                "artifact path changed while it was open"
            ) from exc
        if _unsafe_node(path_after) or not stat.S_ISREG(path_after.st_mode):
            raise ArtifactSecurityError(
                "artifact path changed into a link, reparse point, or special file"
            )
        if (
            _file_identity(opened) is not None
            and _file_identity(path_after) is not None
            and not _same_file(opened, path_after)
        ):
            raise ArtifactSecurityError("artifact path changed while it was open")
        if (
            _file_identity(opened) is None
            or _file_identity(path_after) is None
        ) and not path_revalidated_by_handle:
            raise ArtifactSecurityError(
                "artifact path identity could not be revalidated"
            )
        data = b"".join(chunks)
        budget.add(len(data))
        if expected_size is not None and len(data) != expected_size:
            raise ArtifactIntegrityError("artifact size does not match its descriptor")
        if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ArtifactIntegrityError("artifact content hash mismatch")
        return data

    def _read_json(
        self,
        path: Path,
        *,
        expected: Optional[ArtifactRef] = None,
        budget: Optional[_ReadBudget] = None,
        maximum: Optional[int] = None,
        evaluator_context_policy: Optional[str] = None,
    ) -> Any:
        active_budget = budget or _ReadBudget(self.max_total_read_bytes)
        data = self._read_bytes(
            path,
            expected_sha256=None if expected is None else expected.sha256,
            expected_size=None if expected is None else expected.size_bytes,
            budget=active_budget,
            maximum_bytes=maximum,
        )
        value = _strict_json_loads(
            data,
            min(self.max_file_bytes, maximum or self.max_file_bytes),
            "artifact JSON",
        )
        if canonical_json_bytes(value) != data:
            raise ArtifactIntegrityError("JSON artifact is not canonical UTF-8 JSON")
        validate_safe_json(
            value,
            "artifact",
            evaluator_context_policy=_evaluator_context_policy_for_payload(
                value,
                evaluator_context_policy,
            ),
        )
        return value

    def _read_text(
        self,
        path: Path,
        *,
        expected: Optional[ArtifactRef] = None,
        budget: Optional[_ReadBudget] = None,
        maximum: Optional[int] = None,
        context: str = "report",
        allow_rendered_environment_projection: bool = False,
    ) -> str:
        active_budget = budget or _ReadBudget(self.max_total_read_bytes)
        data = self._read_bytes(
            path,
            expected_sha256=None if expected is None else expected.sha256,
            expected_size=None if expected is None else expected.size_bytes,
            budget=active_budget,
            maximum_bytes=maximum,
        )
        try:
            text = data.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise ArtifactIntegrityError(
                "%s is not canonical UTF-8 text" % context
            ) from exc
        try:
            _validated_artifact_text(
                text,
                context,
                allow_rendered_environment_projection=(
                    allow_rendered_environment_projection
                ),
            )
        except SchemaError as exc:
            raise ArtifactIntegrityError(
                "%s violates the safe text boundary" % context
            ) from exc
        if "\r" in text:
            raise ArtifactIntegrityError(
                "%s must use canonical LF line endings" % context
            )
        return text

    def _artifact_ref(self, run_id: str, path: Path, data: bytes) -> ArtifactRef:
        return ArtifactRef(
            relative_path=self._run_relative(run_id, path),
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    def _write_json(
        self,
        run_id: str,
        relative_path: str,
        value: Any,
        *,
        maximum: Optional[int] = None,
        evaluator_context_policy: Optional[str] = None,
    ) -> ArtifactRef:
        validate_safe_json(
            value,
            "artifact",
            evaluator_context_policy=_evaluator_context_policy_for_payload(
                value,
                evaluator_context_policy,
            ),
        )
        data = canonical_json_bytes(value)
        if len(data) > min(self.max_file_bytes, maximum or self.max_file_bytes):
            raise ArtifactIntegrityError("JSON artifact exceeds its byte limit")
        path = self._target(run_id, relative_path)
        self._write_bytes_exclusive(path, data)
        return self._artifact_ref(run_id, path, data)

    def _write_text(
        self,
        run_id: str,
        relative_path: str,
        value: str,
        *,
        maximum: int,
        allow_rendered_environment_projection: bool = False,
    ) -> ArtifactRef:
        text = _validated_artifact_text(
            value,
            "report",
            allow_rendered_environment_projection=(
                allow_rendered_environment_projection
            ),
        )
        if "\r" in text:
            raise SchemaError("report must use canonical LF line endings")
        data = text.encode("utf-8", "strict")
        if len(data) > min(maximum, self.max_file_bytes):
            raise ArtifactIntegrityError("report exceeds its byte limit")
        path = self._target(run_id, relative_path)
        self._write_bytes_exclusive(path, data)
        return self._artifact_ref(run_id, path, data)

    def _adopt_json(
        self,
        run_id: str,
        relative_path: str,
        *,
        budget: _ReadBudget,
        maximum: int,
    ) -> Tuple[Any, ArtifactRef]:
        path = self._target(run_id, relative_path)
        data = self._read_bytes(
            path,
            expected_sha256=None,
            expected_size=None,
            budget=budget,
            maximum_bytes=maximum,
        )
        value = _strict_json_loads(
            data, min(maximum, self.max_file_bytes), "orphan artifact JSON"
        )
        if canonical_json_bytes(value) != data:
            raise ArtifactIntegrityError("orphan JSON artifact is not canonical")
        validate_safe_json(value, "artifact")
        return value, self._artifact_ref(run_id, path, data)

    def read_json_artifact(self, run_id: str, artifact: ArtifactRef) -> Any:
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("artifact must be an ArtifactRef")
        return self._read_json(
            self._target(run_id, artifact.relative_path), expected=artifact
        )

    def read_json_artifacts(
        self, run_id: str, artifacts: Iterable[ArtifactRef]
    ) -> Tuple[Any, ...]:
        budget = _ReadBudget(self.max_total_read_bytes)
        values: List[Any] = []
        for artifact in artifacts:
            if not isinstance(artifact, ArtifactRef):
                raise TypeError("artifacts must contain ArtifactRef values")
            values.append(
                self._read_json(
                    self._target(run_id, artifact.relative_path),
                    expected=artifact,
                    budget=budget,
                )
            )
        return tuple(values)

    @staticmethod
    def _validate_snapshot_binding(
        config: EvalRunConfig, case_snapshot: RunCaseSnapshot
    ) -> None:
        if not isinstance(config, EvalRunConfig):
            raise TypeError("config must be an EvalRunConfig")
        if not isinstance(case_snapshot, RunCaseSnapshot):
            raise TypeError("case_snapshot must be a RunCaseSnapshot")
        expected_suite = SuiteRunConfig.from_case_snapshot(case_snapshot)
        if config.suite != expected_suite:
            raise SchemaError(
                "Run Config suite does not match the verified RunCaseSnapshot"
            )

    def create_run(
        self,
        config: EvalRunConfig,
        case_snapshot: RunCaseSnapshot,
        *,
        run_preflight: Any = None,
    ) -> RunManifest:
        """Materialize the complete immutable Run/Trial plan, manifest last."""

        self._validate_snapshot_binding(config, case_snapshot)
        config_data = canonical_json_bytes(config)
        snapshot_data = canonical_json_bytes(case_snapshot)
        if len(config_data) > min(self.max_file_bytes, MAX_EVAL_RUN_CONFIG_BYTES):
            raise ArtifactIntegrityError("Run Config exceeds its storage byte limit")
        if len(snapshot_data) > min(
            self.max_file_bytes, MAX_RUN_CASE_SNAPSHOT_BYTES
        ):
            raise ArtifactIntegrityError("Case Snapshot exceeds its storage byte limit")
        preflight_bytes = b""
        if run_preflight is not None:
            validate_safe_json(run_preflight, "capability preflight")
            # The final envelope adds only fixed-size digests/field names.
            # Reserve space before creating the Run directory so an
            # unpersistable preflight cannot strand an immutable Run ID.
            preflight_bytes = canonical_json_bytes(run_preflight)
            effective_limit = min(self.max_file_bytes, MAX_RUN_PREFLIGHT_BYTES)
            if len(preflight_bytes) + 2_048 > effective_limit:
                raise ArtifactIntegrityError(
                    "Run capability preflight cannot fit its receipt envelope"
                )
        config_ref = self._planned_artifact_ref(
            config.run_id, "run_config.json", config_data
        )
        snapshot_ref = self._planned_artifact_ref(
            config.run_id, "case_snapshot.json", snapshot_data
        )
        initial_evaluator_execution_digest = (
            EvaluatorExecutionConfig.from_resource_budgets(
                config.evaluator, config.resource_budgets
            ).digest()
        )
        plans: List[RunTrialPlan] = []
        trial_payloads: List[
            Tuple[str, TrialManifest, bytes, ArtifactRef]
        ] = []
        for case in config.suite.cases:
            case_path_id = derive_case_path_id(case.task_id)
            for trial_index in range(1, config.trial_count + 1):
                trial_id = derive_trial_id(config.run_id, case.task_id, trial_index)
                relative_base = "cases/%s/trials/%s" % (case_path_id, trial_id)
                trial_manifest = TrialManifest(
                    schema_version=TrialManifest.SCHEMA_VERSION,
                    run_id=config.run_id,
                    task_id=case.task_id,
                    case_path_id=case_path_id,
                    canonical_case_digest=case.canonical_case_digest,
                    eval_input_digest=case.eval_input_digest,
                    wire_contract=config.wire_contract,
                    target_kind=config.wire_contract.review_target_kind,
                    materializer_protocol=config.materializer_protocol,
                    suite_preparation_binding_digest=(
                        config.suite_preparation_binding_digest
                    ),
                    adapter_capabilities_digest=(
                        config.adapter_capabilities_digest
                    ),
                    trial_id=trial_id,
                    trial_index=trial_index,
                    seed=derive_trial_seed(config.run_id, case.task_id, trial_index),
                    agent_config_digest=config.agent_config_digest,
                    initial_evaluator_execution_digest=(
                        initial_evaluator_execution_digest
                    ),
                )
                trial_data = canonical_json_bytes(trial_manifest)
                if len(trial_data) > min(
                    self.max_file_bytes, MAX_TRIAL_MANIFEST_BYTES
                ):
                    raise ArtifactIntegrityError(
                        "Trial manifest exceeds its storage byte limit"
                    )
                relative_manifest = "%s/trial_manifest.json" % relative_base
                manifest_ref = self._planned_artifact_ref(
                    config.run_id, relative_manifest, trial_data
                )
                trial_payloads.append(
                    (relative_base, trial_manifest, trial_data, manifest_ref)
                )
                plans.append(
                    RunTrialPlan(
                        task_id=case.task_id,
                        case_path_id=case_path_id,
                        canonical_case_digest=case.canonical_case_digest,
                        eval_input_digest=case.eval_input_digest,
                        trial_id=trial_id,
                        trial_index=trial_index,
                        manifest=manifest_ref,
                    )
                )
        manifest = RunManifest(
            schema_version=RunManifest.SCHEMA_VERSION,
            run_id=config.run_id,
            run_config=config_ref,
            case_snapshot=snapshot_ref,
            wire_contract=config.wire_contract,
            suite_preparation_binding_digest=(
                config.suite_preparation_binding_digest
            ),
            adapter_capabilities_digest=config.adapter_capabilities_digest,
            agent_config_digest=config.agent_config_digest,
            initial_evaluator_execution_digest=(
                initial_evaluator_execution_digest
            ),
            trials=tuple(plans),
        )
        manifest_data = canonical_json_bytes(manifest)
        if len(manifest_data) > min(self.max_file_bytes, MAX_RUN_MANIFEST_BYTES):
            raise ArtifactIntegrityError("Run manifest exceeds its storage byte limit")
        read_floor = (
            len(config_data)
            + len(snapshot_data)
            + len(preflight_bytes)
            + len(manifest_data)
            + sum(len(item[2]) for item in trial_payloads)
        )
        if read_floor > self.max_total_read_bytes:
            raise ArtifactIntegrityError(
                "Run control plane exceeds ArtifactStore read capacity"
            )
        run_dir = self._run_dir(config.run_id)
        self._assert_directory(self.root)
        with self._guard_parent_directory(run_dir) as parent_descriptor:
            try:
                if parent_descriptor is None:
                    os.mkdir(run_dir, 0o700)
                else:
                    os.mkdir(
                        run_dir.name,
                        0o700,
                        dir_fd=parent_descriptor,
                    )
                    os.fsync(parent_descriptor)
            except FileExistsError as exc:
                raise ArtifactConflictError(
                    "run instance already exists; use a new run_instance_key"
                ) from exc
            except OSError as exc:
                raise ArtifactSecurityError("could not create run directory") from exc
        self._assert_directory(run_dir)
        for fixed in (".locks", "cases", "evaluations", "receipts"):
            self._ensure_directory(run_dir / fixed)
        config_ref_written = self._write_json(
            config.run_id,
            "run_config.json",
            config,
            maximum=MAX_EVAL_RUN_CONFIG_BYTES,
        )
        snapshot_ref_written = self._write_json(
            config.run_id,
            "case_snapshot.json",
            case_snapshot,
            maximum=MAX_RUN_CASE_SNAPSHOT_BYTES,
        )
        if config_ref_written != config_ref or snapshot_ref_written != snapshot_ref:
            raise ArtifactIntegrityError(
                "Run plan input artifact identity changed before publication"
            )
        for relative_base, trial_manifest, trial_data, manifest_ref in trial_payloads:
            for suffix in (
                ".locks",
                "receipts",
                "evaluations",
                "materializations",
            ):
                self._ensure_directory(
                    self._target(config.run_id, relative_base) / suffix
                )
            written = self._write_json(
                config.run_id,
                "%s/trial_manifest.json" % relative_base,
                trial_manifest,
                maximum=MAX_TRIAL_MANIFEST_BYTES,
            )
            if written != manifest_ref or canonical_json_bytes(trial_manifest) != trial_data:
                raise ArtifactIntegrityError(
                    "Trial manifest identity changed before publication"
                )
        if run_preflight is not None:
            self._write_run_preflight_payload(
                config,
                manifest,
                run_preflight,
            )
        # This is the Run-plan commit marker and is always published last.
        self._write_json(
            config.run_id,
            "run_manifest.json",
            manifest,
            maximum=MAX_RUN_MANIFEST_BYTES,
        )
        return manifest

    def _write_run_preflight_payload(
        self,
        config: EvalRunConfig,
        manifest: RunManifest,
        preflight: Any,
    ) -> ArtifactRef:
        validate_safe_json(preflight, "capability preflight")
        preflight_bytes = canonical_json_bytes(preflight)
        envelope = {
            "schema_version": EVAL_RUN_PREFLIGHT_SCHEMA_VERSION,
            "run_id": config.run_id,
            "run_manifest_digest": manifest.digest(),
            "run_config_digest": config.digest(),
            "wire_contract": config.wire_contract.to_dict(),
            "adapter_capabilities_digest": (
                config.adapter_capabilities_digest
            ),
            "target_kinds": [item.value for item in config.target_kinds],
            "materializer_protocol": config.materializer_protocol,
            "preflight_digest": hashlib.sha256(preflight_bytes).hexdigest(),
            "preflight": preflight,
        }
        validate_safe_json(envelope, "capability preflight envelope")
        return self._write_json(
            config.run_id,
            "receipts/capability_preflight.json",
            envelope,
            maximum=MAX_RUN_PREFLIGHT_BYTES,
        )

    def write_run_preflight(self, run_id: str, preflight: Any) -> ArtifactRef:
        """Persist one hash-bound, Run-level capability preflight receipt.

        The receipt is deliberately separate from Trial terminal artifacts and
        binds the accepted immutable Run manifest/config.  Rejected strict
        candidates use ``write_preflight_candidate`` because no Run manifest
        exists for them.  Neither form is an evaluator artifact or Case truth.
        """

        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        return self._write_run_preflight_payload(
            bundle.config,
            bundle.manifest,
            preflight,
        )

    def write_preflight_candidate(
        self,
        run_id: str,
        *,
        run_config_digest: str,
        case_snapshot_digest: str,
        preflight: Any,
    ) -> ArtifactRef:
        """Persist a rejected candidate before an immutable Run exists."""

        validate_run_id(run_id)
        _digest(run_config_digest, "preflight candidate.run_config_digest")
        _digest(case_snapshot_digest, "preflight candidate.case_snapshot_digest")
        validate_safe_json(preflight, "capability preflight")
        preflight_bytes = canonical_json_bytes(preflight)
        envelope = {
            "schema_version": EVAL_PREFLIGHT_CANDIDATE_SCHEMA_VERSION,
            "run_id": run_id,
            "run_config_digest": run_config_digest,
            "case_snapshot_digest": case_snapshot_digest,
            "preflight_digest": hashlib.sha256(preflight_bytes).hexdigest(),
            "preflight": preflight,
        }
        validate_safe_json(envelope, "preflight candidate envelope")
        data = canonical_json_bytes(envelope)
        if len(data) > min(self.max_file_bytes, MAX_PREFLIGHT_CANDIDATE_BYTES):
            raise ArtifactIntegrityError("preflight candidate exceeds its byte limit")
        self._ensure_directory(self.root / "preflights")
        path = self.root / "preflights" / (run_id + ".json")
        self._write_bytes_exclusive(path, data)
        return ArtifactRef(
            relative_path="preflights/%s.json" % run_id,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    def load_preflight_candidate(self, run_id: str) -> Any:
        """Read a rejected candidate receipt without creating a Run."""

        validate_run_id(run_id)
        budget = _ReadBudget(self.max_total_read_bytes)
        path = self.root / "preflights" / (run_id + ".json")
        payload = self._read_json(
            path,
            budget=budget,
            maximum=MAX_PREFLIGHT_CANDIDATE_BYTES,
        )
        if type(payload) is not dict:
            raise ArtifactIntegrityError("preflight candidate is not an object")
        if "schema_version" in payload:
            _require_protocol_version(
                payload["schema_version"],
                EVAL_PREFLIGHT_CANDIDATE_SCHEMA_VERSION,
                "preflight candidate.schema_version",
            )
        expected_fields = {
            "schema_version",
            "run_id",
            "run_config_digest",
            "case_snapshot_digest",
            "preflight_digest",
            "preflight",
        }
        if (
            set(payload) != expected_fields
            or payload.get("run_id") != run_id
            or "preflight" not in payload
        ):
            raise ArtifactIntegrityError("preflight candidate binding is invalid")
        try:
            _digest(
                payload.get("run_config_digest"),
                "preflight candidate.run_config_digest",
            )
            _digest(
                payload.get("case_snapshot_digest"),
                "preflight candidate.case_snapshot_digest",
            )
        except SchemaError as exc:
            raise ArtifactIntegrityError(
                "preflight candidate digests are invalid"
            ) from exc
        digest = hashlib.sha256(
            canonical_json_bytes(payload["preflight"])
        ).hexdigest()
        if payload.get("preflight_digest") != digest:
            raise ArtifactIntegrityError("preflight candidate digest is invalid")
        return payload

    def load_run_preflight(self, run_id: str) -> Any:
        """Read and verify the immutable Run-level capability receipt."""

        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        payload = self._read_json(
            self._run_dir(run_id) / "receipts/capability_preflight.json",
            budget=budget,
            maximum=MAX_RUN_PREFLIGHT_BYTES,
        )
        if not isinstance(payload, dict):
            raise ArtifactIntegrityError("Run capability preflight is not an object")
        if "schema_version" in payload:
            _require_protocol_version(
                payload["schema_version"],
                EVAL_RUN_PREFLIGHT_SCHEMA_VERSION,
                "capability preflight.schema_version",
            )
        expected_fields = {
            "schema_version",
            "run_id",
            "run_manifest_digest",
            "run_config_digest",
            "wire_contract",
            "adapter_capabilities_digest",
            "target_kinds",
            "materializer_protocol",
            "preflight_digest",
            "preflight",
        }
        if (
            set(payload) != expected_fields
            or payload.get("run_id") != run_id
            or payload.get("run_manifest_digest") != bundle.manifest.digest()
            or payload.get("run_config_digest") != bundle.config.digest()
            or payload.get("wire_contract")
            != bundle.config.wire_contract.to_dict()
            or payload.get("adapter_capabilities_digest")
            != bundle.config.adapter_capabilities_digest
            or payload.get("target_kinds")
            != [item.value for item in bundle.config.target_kinds]
            or payload.get("materializer_protocol")
            != bundle.config.materializer_protocol
            or "preflight" not in payload
        ):
            raise ArtifactIntegrityError("Run capability preflight binding is invalid")
        preflight_bytes = canonical_json_bytes(payload["preflight"])
        if payload.get("preflight_digest") != hashlib.sha256(preflight_bytes).hexdigest():
            raise ArtifactIntegrityError("Run capability preflight digest is invalid")
        return payload["preflight"]

    def _read_run_manifest(
        self, run_id: str, *, budget: _ReadBudget
    ) -> RunManifest:
        validate_run_id(run_id)
        payload = self._read_json(
            self._run_dir(run_id) / "run_manifest.json",
            budget=budget,
            maximum=MAX_RUN_MANIFEST_BYTES,
        )
        manifest = RunManifest.from_dict(payload)
        if manifest.run_id != run_id:
            raise ArtifactIntegrityError("RunManifest identity does not match its path")
        return manifest

    def _read_run_config(
        self,
        run_id: str,
        manifest: RunManifest,
        *,
        budget: _ReadBudget,
    ) -> EvalRunConfig:
        payload = self._read_json(
            self._run_dir(run_id) / "run_config.json",
            expected=manifest.run_config,
            budget=budget,
            maximum=MAX_EVAL_RUN_CONFIG_BYTES,
        )
        config = EvalRunConfig.from_dict(payload)
        if config.run_id != run_id:
            raise ArtifactIntegrityError("Run Config identity does not match its path")
        if (
            config.agent_config_digest != manifest.agent_config_digest
            or config.wire_contract != manifest.wire_contract
            or config.suite_preparation_binding_digest
            != manifest.suite_preparation_binding_digest
            or config.adapter_capabilities_digest
            != manifest.adapter_capabilities_digest
            or EvaluatorExecutionConfig.from_resource_budgets(
                config.evaluator, config.resource_budgets
            ).digest()
            != manifest.initial_evaluator_execution_digest
        ):
            raise ArtifactIntegrityError("Run Config digests do not match RunManifest")
        return config

    def _read_case_snapshot(
        self,
        run_id: str,
        manifest: RunManifest,
        *,
        budget: _ReadBudget,
    ) -> RunCaseSnapshot:
        payload = self._read_json(
            self._run_dir(run_id) / "case_snapshot.json",
            expected=manifest.case_snapshot,
            budget=budget,
            maximum=MAX_RUN_CASE_SNAPSHOT_BYTES,
        )
        return RunCaseSnapshot.from_dict(payload)

    def _load_verified_run_bundle(
        self, run_id: str, *, budget: Optional[_ReadBudget] = None
    ) -> _VerifiedRunBundle:
        active_budget = budget or _ReadBudget(self.max_total_read_bytes)
        manifest = self._read_run_manifest(run_id, budget=active_budget)
        config = self._read_run_config(
            run_id, manifest, budget=active_budget
        )
        case_snapshot = self._read_case_snapshot(
            run_id, manifest, budget=active_budget
        )
        try:
            self._validate_snapshot_binding(config, case_snapshot)
        except (SchemaError, TypeError) as exc:
            raise ArtifactIntegrityError(
                "Run Config and Case Snapshot bindings do not match"
            ) from exc
        expected_count = len(config.suite.cases) * config.trial_count
        if len(manifest.trials) != expected_count:
            raise ArtifactIntegrityError("RunManifest does not contain the complete Trial plan")
        actual = {
            (item.task_id, item.trial_index): item for item in manifest.trials
        }
        for case in config.suite.cases:
            for trial_index in range(1, config.trial_count + 1):
                item = actual.get((case.task_id, trial_index))
                if item is None or (
                    item.case_path_id != derive_case_path_id(case.task_id)
                    or item.canonical_case_digest
                    != case.canonical_case_digest
                    or item.eval_input_digest != case.eval_input_digest
                    or item.trial_id
                    != derive_trial_id(run_id, case.task_id, trial_index)
                ):
                    raise ArtifactIntegrityError(
                        "RunManifest Trial plan differs from Run Config"
                    )
        return _VerifiedRunBundle(
            manifest=manifest,
            config=config,
            case_snapshot=case_snapshot,
        )

    def load_run_manifest(self, run_id: str) -> RunManifest:
        """Load a Run manifest only after validating its Config and Snapshot."""

        return self._load_verified_run_bundle(run_id).manifest

    def load_run_config(self, run_id: str) -> EvalRunConfig:
        return self._load_verified_run_bundle(run_id).config

    def load_case_snapshot(self, run_id: str) -> RunCaseSnapshot:
        """Load the immutable truth-free Case selection bound by a Run."""

        return self._load_verified_run_bundle(run_id).case_snapshot

    @staticmethod
    def _find_plan(
        manifest: RunManifest, task_id: str, trial_id: str
    ) -> RunTrialPlan:
        for plan in manifest.trials:
            if plan.task_id == task_id and plan.trial_id == trial_id:
                return plan
        raise ArtifactStateError("Trial is not present in immutable RunManifest")

    def _load_trial_manifest(
        self,
        bundle: _VerifiedRunBundle,
        task_id: str,
        trial_id: str,
        *,
        budget: _ReadBudget,
    ) -> TrialManifest:
        run_id = bundle.config.run_id
        plan = self._find_plan(bundle.manifest, task_id, trial_id)
        payload = self._read_json(
            self._target(run_id, plan.manifest.relative_path),
            expected=plan.manifest,
            budget=budget,
            maximum=MAX_TRIAL_MANIFEST_BYTES,
        )
        trial = TrialManifest.from_dict(payload)
        if (
            trial.run_id != run_id
            or trial.task_id != task_id
            or trial.trial_id != trial_id
            or trial.case_path_id != plan.case_path_id
            or trial.canonical_case_digest != plan.canonical_case_digest
            or trial.eval_input_digest != plan.eval_input_digest
            or trial.trial_index != plan.trial_index
            or trial.wire_contract != bundle.config.wire_contract
            or trial.target_kind
            is not bundle.config.wire_contract.review_target_kind
            or trial.materializer_protocol
            != bundle.config.materializer_protocol
            or trial.suite_preparation_binding_digest
            != bundle.config.suite_preparation_binding_digest
            or trial.adapter_capabilities_digest
            != bundle.config.adapter_capabilities_digest
            or trial.agent_config_digest
            != bundle.config.agent_config_digest
            or trial.initial_evaluator_execution_digest
            != EvaluatorExecutionConfig.from_resource_budgets(
                bundle.config.evaluator,
                bundle.config.resource_budgets,
            ).digest()
        ):
            raise ArtifactIntegrityError("TrialManifest does not match RunManifest plan")
        return trial

    def load_trial_manifest(
        self, run_id: str, task_id: str, trial_id: str
    ) -> TrialManifest:
        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        return self._load_trial_manifest(
            bundle, task_id, trial_id, budget=budget
        )

    def create_trial(
        self, run_id: str, task_id: str, trial_index: int
    ) -> TrialManifest:
        """Return a pre-created immutable Trial plan; never append to a manifest."""

        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        bundle.config.suite.case(task_id)
        if trial_index > bundle.config.trial_count:
            raise SchemaError("trial_index exceeds run_config.trial_count")
        trial_id = derive_trial_id(run_id, task_id, trial_index)
        return self._load_trial_manifest(
            bundle, task_id, trial_id, budget=budget
        )

    def _receipt_path(
        self, plan: TrialManifest, stage: StageName, attempt: Optional[int] = None
    ) -> str:
        base = "cases/%s/trials/%s/receipts" % (
            plan.case_path_id,
            plan.trial_id,
        )
        if stage is StageName.PREPARE and attempt is not None:
            return "%s/attempt-%04d/prepare.json" % (base, attempt)
        if stage is StageName.AGENT:
            return "%s/terminal.json" % base
        if stage in {StageName.START, StageName.INCOMPLETE} and attempt is not None:
            return "%s/attempt-%04d/%s.json" % (base, attempt, stage.value)
        raise ArtifactStateError("receipt path requires a legal stage/attempt")

    @staticmethod
    def _materialization_manifest_path(
        plan: TrialManifest, attempt: int
    ) -> str:
        _integer(
            attempt,
            "attempt",
            minimum=1,
            maximum=MAX_TRIAL_ATTEMPTS,
        )
        return (
            "cases/%s/trials/%s/materializations/attempt-%04d/"
            "materialization_manifest.json"
            % (plan.case_path_id, plan.trial_id, attempt)
        )

    def _load_receipt(
        self,
        run_id: str,
        relative_path: str,
        *,
        budget: _ReadBudget,
        expected: Optional[ArtifactRef] = None,
    ) -> StageReceipt:
        payload = self._read_json(
            self._target(run_id, relative_path),
            expected=expected,
            budget=budget,
            maximum=MAX_STAGE_RECEIPT_BYTES,
        )
        return StageReceipt.from_dict(payload)

    def _validate_receipt_binding(
        self,
        receipt: StageReceipt,
        plan: TrialManifest,
        *,
        config_digest: str,
    ) -> None:
        if (
            receipt.run_id != plan.run_id
            or receipt.task_id != plan.task_id
            or receipt.trial_id != plan.trial_id
            or receipt.config_digest != config_digest
        ):
            raise ArtifactIntegrityError("receipt identity/config binding mismatch")
        prefix = "cases/%s/trials/%s/" % (plan.case_path_id, plan.trial_id)
        for artifact in receipt.artifacts:
            if not artifact.relative_path.startswith(prefix):
                raise ArtifactIntegrityError("receipt artifact escapes its Trial namespace")
        if receipt.stage is StageName.PREPARE:
            if receipt.attempt is None:
                raise ArtifactIntegrityError(
                    "prepare receipt has no attempt lease"
                )
            expected_materialization = self._materialization_manifest_path(
                plan, receipt.attempt
            )
            expected_paths = {
                prefix + "input.json",
                expected_materialization,
            }
            if (
                {item.relative_path for item in receipt.artifacts}
                != expected_paths
                or receipt.materialization_manifest is None
                or receipt.materialization_manifest.relative_path
                != expected_materialization
            ):
                raise ArtifactIntegrityError("prepare receipt binds the wrong input path")
        elif receipt.stage is StageName.AGENT:
            submission_paths = [
                item.relative_path
                for item in receipt.artifacts
                if item.relative_path.endswith("/submission.json")
            ]
            if submission_paths != [prefix + "submission.json"]:
                raise ArtifactIntegrityError(
                    "terminal receipt binds the wrong Submission path"
                )
        elif receipt.stage is StageName.EVALUATOR:
            evaluation_prefix = prefix + "evaluations/%s/" % receipt.evaluation_id
            if any(
                not item.relative_path.startswith(evaluation_prefix)
                for item in receipt.artifacts
            ):
                raise ArtifactIntegrityError(
                    "evaluator receipt escapes its evaluation namespace"
                )

    def _verify_receipt(
        self,
        run_id: str,
        receipt: StageReceipt,
        *,
        budget: _ReadBudget,
        maximum_bytes: Optional[int] = None,
    ) -> None:
        for artifact in receipt.artifacts:
            self._read_bytes(
                self._target(run_id, artifact.relative_path),
                expected_sha256=artifact.sha256,
                expected_size=artifact.size_bytes,
                budget=budget,
                maximum_bytes=maximum_bytes,
            )

    def _load_prepare_materialization(
        self,
        bundle: _VerifiedRunBundle,
        plan: TrialManifest,
        receipt: StageReceipt,
        *,
        budget: _ReadBudget,
    ) -> Tuple[EvalInput, TrialMaterializationManifest]:
        if receipt.stage is not StageName.PREPARE or receipt.attempt is None:
            raise ArtifactIntegrityError(
                "prepare receipt has an invalid stage/attempt binding"
            )
        self._validate_receipt_binding(
            receipt,
            plan,
            config_digest=plan.agent_config_digest,
        )
        base = "cases/%s/trials/%s" % (
            plan.case_path_id,
            plan.trial_id,
        )
        expected_input_path = base + "/input.json"
        input_refs = tuple(
            item
            for item in receipt.artifacts
            if item.relative_path == expected_input_path
        )
        if len(input_refs) != 1 or receipt.materialization_manifest is None:
            raise ArtifactIntegrityError(
                "prepare receipt artifact projection is incomplete"
            )
        input_payload = self._read_json(
            self._target(plan.run_id, expected_input_path),
            expected=input_refs[0],
            budget=budget,
            maximum=MAX_EVAL_INPUT_BYTES,
        )
        materialization_payload = self._read_json(
            self._target(
                plan.run_id,
                receipt.materialization_manifest.relative_path,
            ),
            expected=receipt.materialization_manifest,
            budget=budget,
            maximum=MAX_TRIAL_MATERIALIZATION_BYTES,
        )
        try:
            eval_input = EvalInput.from_dict(input_payload)
            materialization = TrialMaterializationManifest.from_dict(
                materialization_payload
            )
        except UnsupportedProtocolVersionError:
            raise
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "prepare receipt child failed strict hydration"
            ) from exc
        if (
            eval_input.task_id != plan.task_id
            or eval_input.digest() != plan.eval_input_digest
            or eval_input.review_target.kind is not plan.target_kind
            or materialization.run_id != plan.run_id
            or materialization.task_id != plan.task_id
            or materialization.trial_id != plan.trial_id
            or materialization.attempt != receipt.attempt
            or materialization.eval_input_digest != eval_input.digest()
            or materialization.review_target_digest
            != eval_input.review_target.digest()
            or materialization.wire_contract != plan.wire_contract
            or materialization.suite_preparation_binding_digest
            != plan.suite_preparation_binding_digest
            or materialization.adapter_capabilities_digest
            != plan.adapter_capabilities_digest
        ):
            raise ArtifactIntegrityError(
                "prepare materialization differs from immutable Run/Trial plan"
            )
        if (
            receipt.materialization_manifest_digest
            != receipt.materialization_manifest.sha256
            or receipt.materialization_id
            != materialization.materialization_id
            or receipt.eval_input_digest != materialization.eval_input_digest
            or receipt.review_target_digest
            != materialization.review_target_digest
            or receipt.prepared_source_id
            != materialization.prepared_source_id
            or receipt.agent_visible_files != materialization.files
            or receipt.adapter_capabilities_digest
            != materialization.adapter_capabilities_digest
            or receipt.target_access != materialization.target_access
        ):
            raise ArtifactIntegrityError(
                "prepare receipt projection differs from materialization manifest"
            )
        return eval_input, materialization

    def _verified_prepare_for_terminal(
        self,
        bundle: _VerifiedRunBundle,
        plan: TrialManifest,
        *,
        attempt: int,
        candidate: Optional[StageReceipt],
        budget: _ReadBudget,
    ) -> Tuple[
        Optional[StageReceipt],
        Optional[EvalInput],
        Optional[TrialMaterializationManifest],
    ]:
        """Resolve and fully verify the Prepare binding for one active attempt."""

        prepare_path = self._receipt_path(plan, StageName.PREPARE, attempt)
        committed: Optional[StageReceipt] = None
        if self._exists_regular(self._target(plan.run_id, prepare_path)):
            committed = self._load_receipt(
                plan.run_id,
                prepare_path,
                budget=budget,
            )
        if candidate is not None and not isinstance(candidate, StageReceipt):
            raise ArtifactIntegrityError(
                "terminal prepare candidate is not a StageReceipt"
            )
        if committed is not None and candidate is not None and committed != candidate:
            raise ArtifactIntegrityError(
                "terminal prepare candidate differs from committed receipt"
            )
        prepare = committed if committed is not None else candidate
        if prepare is None:
            materialization_path = self._materialization_manifest_path(
                plan,
                attempt,
            )
            if self._exists_regular(
                self._target(plan.run_id, materialization_path)
            ):
                raise ArtifactIntegrityError(
                    "materialization exists without a committed prepare receipt"
                )
            return None, None, None
        if prepare.stage is not StageName.PREPARE or prepare.attempt != attempt:
            raise ArtifactIntegrityError(
                "terminal prepare receipt does not bind the active attempt"
            )
        eval_input, materialization = self._load_prepare_materialization(
            bundle,
            plan,
            prepare,
            budget=budget,
        )
        return prepare, eval_input, materialization

    def _terminal_target_binding(
        self,
        bundle: _VerifiedRunBundle,
        plan: TrialManifest,
        *,
        attempt: int,
        prepare: Optional[StageReceipt],
        budget: _ReadBudget,
    ) -> Tuple[str, bool]:
        _prepare, _eval_input, materialization = self._verified_prepare_for_terminal(
            bundle,
            plan,
            attempt=attempt,
            candidate=prepare,
            budget=budget,
        )
        if materialization is not None:
            return materialization.materialization_id, True
        eval_input = bundle.case_snapshot.eval_input(plan.task_id)
        if eval_input.digest() != plan.eval_input_digest:
            raise ArtifactIntegrityError(
                "Case Snapshot EvalInput differs from immutable Trial plan"
            )
        return (
            derive_pre_materialization_failure_binding(
                run_id=plan.run_id,
                task_id=plan.task_id,
                trial_id=plan.trial_id,
                attempt=attempt,
                eval_input_digest=plan.eval_input_digest,
                review_target_digest=eval_input.review_target.digest(),
            ),
            False,
        )

    def _verify_existing_terminal_input(
        self,
        bundle: _VerifiedRunBundle,
        plan: TrialManifest,
        *,
        budget: _ReadBudget,
    ) -> None:
        input_path = "cases/%s/trials/%s/input.json" % (
            plan.case_path_id,
            plan.trial_id,
        )
        target = self._target(plan.run_id, input_path)
        if not self._exists_regular(target):
            return
        payload = self._read_json(
            target,
            budget=budget,
            maximum=MAX_EVAL_INPUT_BYTES,
        )
        try:
            eval_input = EvalInput.from_dict(payload)
        except UnsupportedProtocolVersionError:
            raise
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "terminal EvalInput failed strict hydration"
            ) from exc
        if (
            eval_input.task_id != plan.task_id
            or eval_input.digest() != plan.eval_input_digest
            or eval_input.review_target.kind is not plan.target_kind
        ):
            raise ArtifactIntegrityError(
                "terminal EvalInput differs from immutable Trial plan"
            )

    def _validate_terminal_submission_binding(
        self,
        bundle: _VerifiedRunBundle,
        plan: TrialManifest,
        submission: EvalSubmission,
        *,
        attempt: int,
        prepare: Optional[StageReceipt],
        budget: _ReadBudget,
        error_type: type[Exception] = ArtifactIntegrityError,
    ) -> None:
        """Enforce one terminal binding matrix before publication or adoption."""

        def reject(message: str) -> None:
            raise error_type(message)

        if (
            submission.task_id != plan.task_id
            or submission.trial_id != plan.trial_id
            or submission.agent_id != bundle.config.agent.agent_id
        ):
            reject("Submission identity does not match immutable Run/Trial plan")
        if submission.eval_input_digest != plan.eval_input_digest:
            reject("Submission EvalInput digest differs from immutable Trial plan")

        target_binding, has_prepare = self._terminal_target_binding(
            bundle,
            plan,
            attempt=attempt,
            prepare=prepare,
            budget=budget,
        )
        if not has_prepare:
            self._verify_existing_terminal_input(
                bundle,
                plan,
                budget=budget,
            )
        failure_code = (
            None if submission.failure is None else submission.failure.code
        )
        if failure_code is FailureCode.HARNESS_MATERIALIZATION_ERROR and (
            submission.failure is None
            or submission.failure.retryable
            or submission.trace_ref is not None
            or submission.usage.input_tokens is not None
            or submission.usage.output_tokens is not None
            or submission.usage.total_tokens is not None
            or submission.usage.tool_calls is not None
            or submission.usage.cost_amount is not None
            or submission.usage.cost_currency is not None
        ):
            reject(
                "Harness-owned materialization failure contains Agent-owned metadata"
            )
        if has_prepare:
            if submission.target_materialization_id != target_binding:
                reject(
                    "Submission target differs from active materialization binding"
                )
            if failure_code is FailureCode.HARNESS_MATERIALIZATION_ERROR and (
                submission.status is not SubmissionStatus.FAILED
                or submission.intent is not None
                or submission.review is not None
                or submission.evidence
            ):
                reject(
                    "post-Prepare materialization failure contains Agent output"
                )
            return

        if (
            submission.status is not SubmissionStatus.FAILED
            or failure_code is not FailureCode.HARNESS_MATERIALIZATION_ERROR
            or submission.intent is not None
            or submission.review is not None
            or submission.evidence
        ):
            reject(
                "terminal Submission requires a committed prepare receipt"
            )
        if submission.target_materialization_id != target_binding:
            reject(
                "Submission pre-materialization failure binding is not canonical"
            )

    def _load_terminal_submission(
        self,
        bundle: _VerifiedRunBundle,
        plan: TrialManifest,
        receipt: StageReceipt,
        prepare: Optional[StageReceipt],
        *,
        budget: _ReadBudget,
    ) -> EvalSubmission:
        expected_path = "cases/%s/trials/%s/submission.json" % (
            plan.case_path_id,
            plan.trial_id,
        )
        submission_refs = [
            artifact
            for artifact in receipt.artifacts
            if artifact.relative_path == expected_path
        ]
        if len(submission_refs) != 1:
            raise ArtifactIntegrityError(
                "terminal receipt must uniquely bind one Submission"
            )
        submission_ref = submission_refs[0]
        for artifact in receipt.artifacts:
            if artifact is submission_ref:
                continue
            artifact_name = artifact.relative_path.rsplit("/", 1)[-1]
            execution_runner = (
                "/runner/" in artifact.relative_path
                and artifact_name not in _CONTROL_RUNNER_ARTIFACT_NAMES
            )
            self._read_bytes(
                self._target(plan.run_id, artifact.relative_path),
                expected_sha256=artifact.sha256,
                expected_size=artifact.size_bytes,
                budget=budget,
                maximum_bytes=(
                    bundle.config.resource_budgets.max_execution_artifact_file_bytes
                    if execution_runner
                    else self.max_file_bytes
                ),
            )
        payload = self._read_json(
            self._target(plan.run_id, submission_ref.relative_path),
            expected=submission_ref,
            budget=budget,
            maximum=MAX_EVAL_SUBMISSION_BYTES,
        )
        submission = EvalSubmission.from_dict(payload)
        runner_names = {
            artifact.relative_path.rsplit("/", 1)[-1]
            for artifact in receipt.artifacts
            if "/runner/" in artifact.relative_path
        }
        missing_required = _required_runner_artifact_names(submission).difference(
            runner_names
        )
        if missing_required:
            raise ArtifactIntegrityError(
                "terminal receipt lacks required Runner artifacts: %s"
                % sorted(missing_required)
            )
        if receipt.terminal_status is None:
            raise ArtifactIntegrityError("terminal receipt lacks terminal status")
        expected_failure = (
            None if submission.failure is None else submission.failure.code
        )
        if (
            submission.status.value != receipt.terminal_status.value
            or expected_failure is not receipt.failure_code
        ):
            raise ArtifactIntegrityError(
                "Submission does not match its terminal receipt"
            )
        if receipt.attempt is None:
            raise ArtifactIntegrityError(
                "terminal receipt has no active attempt binding"
            )
        self._validate_terminal_submission_binding(
            bundle,
            plan,
            submission,
            attempt=receipt.attempt,
            prepare=prepare,
            budget=budget,
        )
        return submission

    def _recoverable_runner_artifacts(
        self,
        run_id: str,
        plan: TrialManifest,
        *,
        attempt: int,
        budget: _ReadBudget,
        maximum_bytes: int,
    ) -> Tuple[ArtifactRef, ...]:
        """Return only known, canonical Runner metadata left by an interrupted write."""

        runner_root = self._trial_dir(plan) / "runner"
        if not os.path.lexists(runner_root):
            return ()
        self._assert_directory(runner_root)
        runner_dir = runner_root / ("attempt-%04d" % attempt)
        if not os.path.lexists(runner_dir):
            return ()
        self._assert_directory(runner_dir)
        refs: List[ArtifactRef] = []
        try:
            entries = sorted(os.scandir(runner_dir), key=lambda item: item.name)
        except OSError as exc:
            raise ArtifactSecurityError(
                "could not inspect interrupted Runner artifacts"
            ) from exc
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArtifactSecurityError(
                    "could not inspect interrupted Runner artifact"
                ) from exc
            if _unsafe_node(info) or not stat.S_ISREG(info.st_mode):
                raise ArtifactSecurityError(
                    "Runner artifact tree contains an unsafe filesystem node"
                )
            name = _runner_artifact_name(entry.name)
            path = runner_dir / name
            artifact_maximum = (
                self.max_file_bytes
                if name in _CONTROL_RUNNER_ARTIFACT_NAMES
                else maximum_bytes
            )
            data = self._read_bytes(
                path,
                expected_sha256=None,
                expected_size=None,
                budget=budget,
                maximum_bytes=artifact_maximum,
            )
            value = _strict_json_loads(
                data,
                min(self.max_file_bytes, artifact_maximum),
                "Runner artifact JSON",
            )
            if canonical_json_bytes(value) != data:
                raise ArtifactIntegrityError(
                    "interrupted Runner artifact is not canonical JSON"
                )
            validate_safe_json(value, "Runner artifact")
            refs.append(self._artifact_ref(run_id, path, data))
        return tuple(refs)

    def _attempt_indices(self, plan: TrialManifest) -> Tuple[int, ...]:
        receipts = self._trial_dir(plan) / "receipts"
        self._assert_directory(receipts)
        indices: List[int] = []
        try:
            entries = list(os.scandir(receipts))
        except OSError as exc:
            raise ArtifactSecurityError(
                "could not inspect Trial attempt receipts"
            ) from exc
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArtifactSecurityError(
                    "could not inspect Trial attempt receipts"
                ) from exc
            if _unsafe_node(metadata):
                raise ArtifactSecurityError(
                    "Trial receipts contain a link or reparse point"
                )
            if stat.S_ISDIR(metadata.st_mode):
                if _ATTEMPT_RE.fullmatch(entry.name) is None:
                    raise ArtifactIntegrityError(
                        "Trial receipts contain an unknown directory"
                    )
                index = int(entry.name.split("-", 1)[1])
                if index < 1 or index > MAX_TRIAL_ATTEMPTS:
                    raise ArtifactIntegrityError(
                        "Trial receipt attempt index is out of range"
                    )
                indices.append(index)
            elif not stat.S_ISREG(metadata.st_mode):
                raise ArtifactSecurityError(
                    "Trial receipts contain a special file"
                )
            elif (
                entry.name != "terminal.json"
                and _TEMP_FILE_RE.fullmatch(entry.name) is None
            ):
                raise ArtifactIntegrityError(
                    "Trial receipts contain an unknown file"
                )
        ordered = tuple(sorted(indices))
        if ordered and ordered != tuple(range(1, ordered[-1] + 1)):
            raise ArtifactIntegrityError(
                "Trial attempt receipt directories are not contiguous"
            )
        return ordered

    def _load_trial_state(
        self,
        bundle: _VerifiedRunBundle,
        plan: TrialManifest,
        *,
        budget: _ReadBudget,
    ) -> Tuple[TrialState, Optional[EvalSubmission]]:
        terminal: Optional[StageReceipt] = None
        terminal_path = self._receipt_path(plan, StageName.AGENT)
        if self._exists_regular(self._target(plan.run_id, terminal_path)):
            terminal = self._load_receipt(
                plan.run_id, terminal_path, budget=budget
            )
            self._validate_receipt_binding(
                terminal, plan, config_digest=plan.agent_config_digest
            )
            if terminal.stage is not StageName.AGENT:
                raise ArtifactIntegrityError("terminal receipt has the wrong stage")

        completed: List[StageName] = []
        attempt_indices = self._attempt_indices(plan)
        latest_attempt: Optional[int] = None
        latest_incomplete = False
        latest_prepare: Optional[StageReceipt] = None
        for attempt in attempt_indices:
            start_path = self._receipt_path(plan, StageName.START, attempt)
            incomplete_path = self._receipt_path(plan, StageName.INCOMPLETE, attempt)
            prepare_path = self._receipt_path(plan, StageName.PREPARE, attempt)
            has_start = self._exists_regular(self._target(plan.run_id, start_path))
            has_incomplete = self._exists_regular(
                self._target(plan.run_id, incomplete_path)
            )
            has_prepare = self._exists_regular(
                self._target(plan.run_id, prepare_path)
            )
            if not has_start:
                if has_incomplete:
                    raise ArtifactIntegrityError(
                        "incomplete receipt exists without matching start receipt"
                    )
                raise ArtifactIntegrityError(
                    "Trial attempt directory has no start receipt"
                )
            start = self._load_receipt(plan.run_id, start_path, budget=budget)
            self._validate_receipt_binding(
                start, plan, config_digest=plan.agent_config_digest
            )
            if start.stage is not StageName.START or start.attempt != attempt:
                raise ArtifactIntegrityError("start receipt has invalid attempt binding")
            latest_attempt = attempt
            latest_incomplete = has_incomplete
            latest_prepare = None
            if has_prepare:
                prepare = self._load_receipt(
                    plan.run_id, prepare_path, budget=budget
                )
                if (
                    prepare.stage is not StageName.PREPARE
                    or prepare.attempt != attempt
                ):
                    raise ArtifactIntegrityError(
                        "prepare receipt has invalid attempt binding"
                    )
                self._load_prepare_materialization(
                    bundle,
                    plan,
                    prepare,
                    budget=budget,
                )
                latest_prepare = prepare
            if has_incomplete:
                incomplete = self._load_receipt(
                    plan.run_id, incomplete_path, budget=budget
                )
                self._validate_receipt_binding(
                    incomplete, plan, config_digest=plan.agent_config_digest
                )
                if (
                    incomplete.stage is not StageName.INCOMPLETE
                    or incomplete.attempt != attempt
                ):
                    raise ArtifactIntegrityError(
                        "incomplete receipt has invalid attempt binding"
                    )
            elif attempt < attempt_indices[-1]:
                next_start = self._receipt_path(plan, StageName.START, attempt + 1)
                if self._exists_regular(self._target(plan.run_id, next_start)):
                    raise ArtifactIntegrityError(
                        "new attempt starts before prior attempt is incomplete"
                    )

        if latest_prepare is not None:
            completed.append(StageName.PREPARE)
        terminal_submission: Optional[EvalSubmission] = None
        if terminal is not None:
            if terminal.attempt is None or latest_attempt is None:
                raise ArtifactIntegrityError("terminal receipt has no started attempt")
            if terminal.attempt != latest_attempt:
                raise ArtifactIntegrityError("terminal receipt binds the wrong attempt")
            status = terminal.terminal_status
            if status is None:
                raise ArtifactIntegrityError("terminal receipt lacks terminal status")
            if (
                status is TrialStatus.COMPLETED
                and StageName.PREPARE not in completed
            ):
                raise ArtifactIntegrityError(
                    "completed terminal receipt lacks committed prepare stage"
                )
            terminal_submission = self._load_terminal_submission(
                bundle,
                plan,
                terminal,
                latest_prepare,
                budget=budget,
            )
            completed.append(StageName.AGENT)
        elif latest_attempt is None:
            status = TrialStatus.PENDING
        elif latest_incomplete:
            status = TrialStatus.INCOMPLETE
        else:
            status = TrialStatus.RUNNING
        return (
            TrialState(
                trial_id=plan.trial_id,
                status=status,
                active_attempt=latest_attempt,
                next_attempt=1 if latest_attempt is None else latest_attempt + 1,
                completed_stages=tuple(completed),
                terminal_receipt=terminal,
            ),
            terminal_submission,
        )

    def load_trial_state(
        self, run_id: str, task_id: str, trial_id: str
    ) -> TrialState:
        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        plan = self._load_trial_manifest(
            bundle, task_id, trial_id, budget=budget
        )
        state, _submission = self._load_trial_state(
            bundle, plan, budget=budget
        )
        return state

    def load_trial_materialization(
        self,
        run_id: str,
        task_id: str,
        trial_id: str,
    ) -> VerifiedTrialMaterialization:
        """Load the active committed PREPARE projection through one trust root.

        A pre-materialization terminal failure intentionally has no value at
        this boundary and raises ``ArtifactStateError``.  Evaluator callers
        must therefore request it only when a Review replay is required.
        """

        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        plan = self._load_trial_manifest(
            bundle,
            task_id,
            trial_id,
            budget=budget,
        )
        state, _submission = self._load_trial_state(
            bundle,
            plan,
            budget=budget,
        )
        attempt = state.active_attempt
        if attempt is None:
            raise ArtifactStateError("Trial has no active materialization attempt")
        receipt, eval_input, materialization = self._verified_prepare_for_terminal(
            bundle,
            plan,
            attempt=attempt,
            candidate=None,
            budget=budget,
        )
        if receipt is None or eval_input is None or materialization is None:
            raise ArtifactStateError(
                "Trial active attempt has no committed PREPARE materialization"
            )
        snapshot_input = bundle.case_snapshot.eval_input(task_id)
        preparation = bundle.case_snapshot.manifest.source.preparation_binding
        preparation_digest = (
            None if preparation is None else preparation.digest()
        )
        if (
            eval_input != snapshot_input
            or preparation_digest
            != materialization.suite_preparation_binding_digest
        ):
            raise ArtifactIntegrityError(
                "PREPARE input or Suite preparation differs from verified Run bundle"
            )
        return VerifiedTrialMaterialization(
            eval_input=eval_input,
            manifest=materialization,
            trial_manifest=plan,
            prepare_receipt=receipt,
            active_attempt=attempt,
            suite_preparation_binding=preparation,
        )

    def load_run_state(self, run_id: str) -> RunState:
        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        states: List[TrialState] = []
        for entry in bundle.manifest.trials:
            plan = self._load_trial_manifest(
                bundle, entry.task_id, entry.trial_id, budget=budget
            )
            state, _submission = self._load_trial_state(
                bundle, plan, budget=budget
            )
            states.append(state)
        if all(state.status in _TERMINAL_STATUSES for state in states):
            status = RunStatus.COMPLETED
        elif any(state.status is TrialStatus.RUNNING for state in states):
            status = RunStatus.RUNNING
        elif any(state.status is TrialStatus.INCOMPLETE for state in states):
            status = RunStatus.INCOMPLETE
        else:
            status = RunStatus.PENDING
        return RunState(run_id=run_id, status=status, trials=tuple(states))

    def _trial_lock_path(self, plan: TrialManifest) -> Path:
        return self._trial_dir(plan) / ".locks" / "trial.lock"

    def start_trial(self, run_id: str, task_id: str, trial_id: str) -> TrialState:
        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        plan = self._load_trial_manifest(
            bundle, task_id, trial_id, budget=budget
        )
        with self._lock(self._trial_lock_path(plan)):
            state, _submission = self._load_trial_state(
                bundle,
                plan,
                budget=_ReadBudget(self.max_total_read_bytes),
            )
            if state.status not in {TrialStatus.PENDING, TrialStatus.INCOMPLETE}:
                raise ArtifactConflictError("Trial cannot start from its current state")
            if state.next_attempt > MAX_TRIAL_ATTEMPTS:
                raise ArtifactStateError("Trial attempt limit has been reached")
            receipt = StageReceipt.create(
                run_id=run_id,
                task_id=task_id,
                trial_id=trial_id,
                stage=StageName.START,
                config_digest=plan.agent_config_digest,
                attempt=state.next_attempt,
            )
            self._write_json(
                run_id,
                self._receipt_path(plan, StageName.START, state.next_attempt),
                receipt,
                maximum=MAX_STAGE_RECEIPT_BYTES,
            )
        return self.load_trial_state(run_id, task_id, trial_id)

    def mark_trial_incomplete(
        self,
        run_id: str,
        task_id: str,
        trial_id: str,
        *,
        attempt: int,
    ) -> TrialState:
        _integer(attempt, "attempt", minimum=1, maximum=MAX_TRIAL_ATTEMPTS)
        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        plan = self._load_trial_manifest(
            bundle, task_id, trial_id, budget=budget
        )
        with self._lock(self._trial_lock_path(plan)):
            state, _submission = self._load_trial_state(
                bundle,
                plan,
                budget=_ReadBudget(self.max_total_read_bytes),
            )
            if (
                state.status is not TrialStatus.RUNNING
                or state.active_attempt != attempt
            ):
                raise ArtifactStateError("only an active running Trial can become incomplete")
            receipt = StageReceipt.create(
                run_id=run_id,
                task_id=task_id,
                trial_id=trial_id,
                stage=StageName.INCOMPLETE,
                config_digest=plan.agent_config_digest,
                attempt=attempt,
            )
            self._write_json(
                run_id,
                self._receipt_path(
                    plan, StageName.INCOMPLETE, attempt
                ),
                receipt,
                maximum=MAX_STAGE_RECEIPT_BYTES,
            )
        return self.load_trial_state(run_id, task_id, trial_id)

    def write_prepare_stage(
        self,
        run_id: str,
        task_id: str,
        trial_id: str,
        eval_input: EvalInput,
        materialization: TrialMaterializationManifest,
        *,
        attempt: int,
    ) -> StageReceipt:
        if not isinstance(eval_input, EvalInput):
            raise TypeError("eval_input must be an EvalInput")
        if not isinstance(materialization, TrialMaterializationManifest):
            raise TypeError(
                "materialization must be a TrialMaterializationManifest"
            )
        _integer(attempt, "attempt", minimum=1, maximum=MAX_TRIAL_ATTEMPTS)
        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        plan = self._load_trial_manifest(
            bundle, task_id, trial_id, budget=budget
        )
        with self._lock(self._trial_lock_path(plan)):
            state, _submission = self._load_trial_state(
                bundle,
                plan,
                budget=_ReadBudget(self.max_total_read_bytes),
            )
            if (
                state.status is not TrialStatus.RUNNING
                or state.active_attempt != attempt
            ):
                raise ArtifactStateError("prepare stage requires a running Trial")
            if StageName.PREPARE in state.completed_stages:
                raise ArtifactConflictError("prepare stage is already committed")
            if eval_input.task_id != task_id:
                raise SchemaError("EvalInput task_id does not match Trial plan")
            if eval_input.digest() != plan.eval_input_digest:
                raise SchemaError("EvalInput digest does not match immutable Trial plan")
            if eval_input.review_target.kind is not plan.target_kind:
                raise SchemaError(
                    "EvalInput target kind does not match immutable Trial plan"
                )
            if (
                materialization.run_id != run_id
                or materialization.task_id != task_id
                or materialization.trial_id != trial_id
                or materialization.attempt != attempt
            ):
                raise SchemaError(
                    "materialization identity does not match active Trial attempt"
                )
            if (
                materialization.eval_input_digest != plan.eval_input_digest
                or materialization.eval_input_digest != eval_input.digest()
                or materialization.review_target_digest
                != eval_input.review_target.digest()
            ):
                raise SchemaError(
                    "materialization input/target content binding drift"
                )
            if (
                materialization.wire_contract != plan.wire_contract
                or materialization.wire_contract != bundle.config.wire_contract
                or materialization.wire_contract.review_target_kind
                is not eval_input.review_target.kind
            ):
                raise SchemaError(
                    "materialization wire/target binding drift"
                )
            if (
                materialization.suite_preparation_binding_digest
                != plan.suite_preparation_binding_digest
                or materialization.adapter_capabilities_digest
                != plan.adapter_capabilities_digest
            ):
                raise SchemaError(
                    "materialization preparation/capability binding drift"
                )
            base = "cases/%s/trials/%s" % (plan.case_path_id, trial_id)
            input_path = "%s/input.json" % base
            if self._exists_regular(self._target(run_id, input_path)):
                adopted_input, input_ref = self._adopt_json(
                    run_id,
                    input_path,
                    budget=_ReadBudget(self.max_total_read_bytes),
                    maximum=MAX_EVAL_INPUT_BYTES,
                )
                if EvalInput.from_dict(adopted_input) != eval_input:
                    raise ArtifactIntegrityError(
                        "existing Trial EvalInput differs from immutable plan"
                    )
            else:
                input_ref = self._write_json(
                    run_id,
                    input_path,
                    eval_input,
                    maximum=MAX_EVAL_INPUT_BYTES,
                )
            materialization_path = self._materialization_manifest_path(
                plan, attempt
            )
            self._ensure_directory(
                self._target(run_id, materialization_path).parent
            )
            if self._exists_regular(
                self._target(run_id, materialization_path)
            ):
                adopted_materialization, materialization_ref = self._adopt_json(
                    run_id,
                    materialization_path,
                    budget=_ReadBudget(self.max_total_read_bytes),
                    maximum=MAX_TRIAL_MATERIALIZATION_BYTES,
                )
                if (
                    TrialMaterializationManifest.from_dict(
                        adopted_materialization
                    )
                    != materialization
                ):
                    raise ArtifactIntegrityError(
                        "existing Trial materialization differs from active attempt"
                    )
            else:
                materialization_ref = self._write_json(
                    run_id,
                    materialization_path,
                    materialization,
                    maximum=MAX_TRIAL_MATERIALIZATION_BYTES,
                )
            receipt = StageReceipt.create(
                run_id=run_id,
                task_id=task_id,
                trial_id=trial_id,
                stage=StageName.PREPARE,
                config_digest=plan.agent_config_digest,
                artifacts=(input_ref, materialization_ref),
                attempt=attempt,
                materialization_manifest=materialization_ref,
                materialization_manifest_digest=materialization_ref.sha256,
                materialization_id=materialization.materialization_id,
                eval_input_digest=materialization.eval_input_digest,
                review_target_digest=materialization.review_target_digest,
                prepared_source_id=materialization.prepared_source_id,
                agent_visible_files=materialization.files,
                adapter_capabilities_digest=(
                    materialization.adapter_capabilities_digest
                ),
                target_access=materialization.target_access,
            )
            # Receipt is the stage commit marker and is published last.
            self._write_json(
                run_id,
                self._receipt_path(plan, StageName.PREPARE, attempt),
                receipt,
                maximum=MAX_STAGE_RECEIPT_BYTES,
            )
        return receipt

    @staticmethod
    def _submission_status(submission: EvalSubmission) -> TrialStatus:
        return TrialStatus(submission.status.value)

    @staticmethod
    def _evaluation_artifact_limit(
        evaluator_execution: EvaluatorExecutionConfig,
    ) -> int:
        return evaluator_execution.max_execution_artifact_file_bytes

    def _execution_artifact_total_bytes(self, run_id: str) -> int:
        """Count execution-plane bytes without following links or reparse points."""

        run_dir = self._run_dir(run_id)
        self._assert_directory(run_dir)
        total = 0
        pending = [run_dir]
        while pending:
            directory = pending.pop()
            self._assert_directory(directory)
            try:
                entries = list(os.scandir(directory))
            except OSError as exc:
                raise ArtifactSecurityError(
                    "could not inspect execution artifact usage"
                ) from exc
            for entry in entries:
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise ArtifactSecurityError(
                        "could not inspect execution artifact usage"
                    ) from exc
                if _unsafe_node(metadata):
                    raise ArtifactSecurityError(
                        "execution artifact tree contains a link or reparse point"
                    )
                entry_path = Path(entry.path)
                relative_parts = entry_path.relative_to(run_dir).parts
                if stat.S_ISDIR(metadata.st_mode):
                    if entry.name != ".locks":
                        pending.append(entry_path)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise ArtifactSecurityError(
                        "execution artifact tree contains a special file"
                    )
                is_evaluation_artifact = (
                    "evaluations" in relative_parts and entry.name != "receipt.json"
                )
                is_runner_artifact = (
                    "runner" in relative_parts
                    and entry.name in _KNOWN_RUNNER_ARTIFACT_NAMES
                    and entry.name not in _CONTROL_RUNNER_ARTIFACT_NAMES
                )
                if is_evaluation_artifact or is_runner_artifact:
                    total += metadata.st_size
        return total

    def _validate_execution_artifact_total_precommit(
        self,
        run_id: str,
        *,
        maximum_total_bytes: int,
        mandatory_additional_bytes: int = 0,
        optional_additional_bytes: int = 0,
    ) -> None:
        """Fail before publication when committed plus planned bytes exceed budget."""

        current_bytes = self._execution_artifact_total_bytes(run_id)
        if current_bytes > maximum_total_bytes:
            raise ExecutionArtifactBudgetError(
                "execution artifacts exceed the configured cumulative byte limit"
            )
        if current_bytes + mandatory_additional_bytes > maximum_total_bytes:
            raise RequiredExecutionArtifactBudgetError(
                "required trace exceeds the cumulative execution budget"
            )
        if (
            current_bytes
            + mandatory_additional_bytes
            + optional_additional_bytes
            > maximum_total_bytes
        ):
            raise ExecutionArtifactBudgetError(
                "Runner artifacts exceed the configured cumulative byte limit"
            )

    def _planned_artifact_ref(
        self, run_id: str, relative_path: str, data: bytes
    ) -> ArtifactRef:
        return self._artifact_ref(
            run_id, self._target(run_id, relative_path), data
        )

    def finalize_submission(
        self,
        run_id: str,
        task_id: str,
        trial_id: str,
        submission: EvalSubmission,
        *,
        attempt: int,
        runner_artifacts: Optional[Dict[str, Any]] = None,
    ) -> TrialState:
        return self._finalize_submission(
            run_id,
            task_id,
            trial_id,
            submission,
            attempt=attempt,
            allow_incomplete=False,
            runner_artifacts=runner_artifacts,
        )

    def _finalize_submission(
        self,
        run_id: str,
        task_id: str,
        trial_id: str,
        submission: EvalSubmission,
        *,
        attempt: int,
        allow_incomplete: bool,
        runner_artifacts: Optional[Dict[str, Any]] = None,
    ) -> TrialState:
        if not isinstance(submission, EvalSubmission):
            raise TypeError("submission must be an EvalSubmission")
        normalized_runner_artifacts: Dict[str, Any] = {}
        if runner_artifacts is not None:
            if type(runner_artifacts) is not dict:
                raise TypeError("runner_artifacts must be a dict or None")
            if len(runner_artifacts) > MAX_RUNNER_ARTIFACTS:
                raise ExecutionArtifactBudgetError(
                    "Runner artifact count exceeds its bounded limit"
                )
            for name, value in runner_artifacts.items():
                normalized_name = _runner_artifact_name(name)
                validate_safe_json(value, "Runner artifact")
                normalized_runner_artifacts[normalized_name] = value
        runner_artifacts = normalized_runner_artifacts
        _integer(attempt, "attempt", minimum=1, maximum=MAX_TRIAL_ATTEMPTS)
        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        plan = self._load_trial_manifest(
            bundle, task_id, trial_id, budget=budget
        )
        validate_safe_json(submission, "submission")
        with self._lock(self._trial_lock_path(plan)):
            state, _existing_submission = self._load_trial_state(
                bundle,
                plan,
                budget=_ReadBudget(self.max_total_read_bytes),
            )
            allowed_statuses = {TrialStatus.RUNNING}
            if allow_incomplete:
                allowed_statuses.add(TrialStatus.INCOMPLETE)
            if state.status not in allowed_statuses:
                raise ArtifactConflictError("Trial cannot be finalized from this state")
            if state.active_attempt != attempt:
                raise ArtifactConflictError(
                    "stale Trial attempt cannot commit a terminal Submission"
                )
            self._validate_terminal_submission_binding(
                bundle,
                plan,
                submission,
                attempt=attempt,
                prepare=None,
                budget=_ReadBudget(self.max_total_read_bytes),
                error_type=SchemaError,
            )
            missing_required = _required_runner_artifact_names(
                submission
            ).difference(runner_artifacts)
            if missing_required:
                raise ArtifactIntegrityError(
                    "terminal Submission lacks required Runner artifacts: %s"
                    % sorted(missing_required)
                )
            terminal_status = self._submission_status(submission)
            base = "cases/%s/trials/%s" % (plan.case_path_id, trial_id)
            planned_control: List[Tuple[str, Any, bytes, ArtifactRef]] = []
            planned_mandatory: List[Tuple[str, Any, bytes, ArtifactRef]] = []
            planned_optional: List[Tuple[str, Any, bytes, ArtifactRef]] = []
            execution_limit = min(
                self.max_file_bytes,
                bundle.config.resource_budgets.max_execution_artifact_file_bytes,
            )
            for name, value in (runner_artifacts or {}).items():
                relative_path = "%s/runner/attempt-%04d/%s" % (
                    base,
                    attempt,
                    name,
                )
                data = canonical_json_bytes(value)
                control = name in _CONTROL_RUNNER_ARTIFACT_NAMES
                mandatory = name in _MANDATORY_EXECUTION_RUNNER_ARTIFACT_NAMES
                artifact_limit = self.max_file_bytes if control else execution_limit
                if len(data) > artifact_limit:
                    if control:
                        raise ArtifactIntegrityError(
                            "required Runner artifact exceeds its control-plane byte limit"
                        )
                    if mandatory:
                        raise RequiredExecutionArtifactBudgetError(
                            "required trace exceeds its execution byte limit"
                        )
                    raise ExecutionArtifactBudgetError(
                        "Runner artifact exceeds its execution byte limit"
                    )
                planned = (
                    relative_path,
                    value,
                    data,
                    self._planned_artifact_ref(run_id, relative_path, data),
                )
                if control:
                    planned_control.append(planned)
                elif mandatory:
                    planned_mandatory.append(planned)
                else:
                    planned_optional.append(planned)

            submission_path = "%s/submission.json" % base
            submission_data = canonical_json_bytes(submission)
            if len(submission_data) > min(
                self.max_file_bytes,
                MAX_EVAL_SUBMISSION_BYTES,
            ):
                raise ArtifactIntegrityError(
                    "Submission exceeds its control-plane byte limit"
                )
            submission_ref = self._planned_artifact_ref(
                run_id,
                submission_path,
                submission_data,
            )
            trace_plan: Optional[Tuple[str, Any, bytes, ArtifactRef]] = None
            artifacts: List[ArtifactRef] = [submission_ref]
            artifacts.extend(item[3] for item in planned_control)
            artifacts.extend(item[3] for item in planned_mandatory)
            artifacts.extend(item[3] for item in planned_optional)
            if submission.trace_ref is not None:
                trace_path = "%s/trace_ref.json" % base
                trace_data = canonical_json_bytes(submission.trace_ref)
                if len(trace_data) > self.max_file_bytes:
                    raise ArtifactIntegrityError(
                        "trace_ref exceeds its control-plane byte limit"
                    )
                trace_ref = self._planned_artifact_ref(
                    run_id,
                    trace_path,
                    trace_data,
                )
                trace_plan = (
                    trace_path,
                    submission.trace_ref,
                    trace_data,
                    trace_ref,
                )
                artifacts.append(trace_ref)
            receipt = StageReceipt.create(
                run_id=run_id,
                task_id=task_id,
                trial_id=trial_id,
                stage=StageName.AGENT,
                config_digest=plan.agent_config_digest,
                artifacts=artifacts,
                attempt=attempt,
                terminal_status=terminal_status,
                failure_code=(
                    None if submission.failure is None else submission.failure.code
                ),
            )
            receipt_data = canonical_json_bytes(receipt)
            if len(receipt_data) > min(
                self.max_file_bytes,
                MAX_STAGE_RECEIPT_BYTES,
            ):
                raise ArtifactIntegrityError(
                    "terminal receipt exceeds its control-plane byte limit"
                )
            receipt_path = self._receipt_path(plan, StageName.AGENT)
            receipt_ref = self._planned_artifact_ref(
                run_id,
                receipt_path,
                receipt_data,
            )

            # Optional Runner metadata is execution-plane data.  Reserve its
            # missing bytes under the run-wide lock so parallel Trials cannot
            # exceed the cumulative budget.  Required clarification/terminal/
            # trace audit artifacts follow control-plane limits and cannot be
            # dropped by an optional execution budget.  Local trace capture is
            # mandatory execution data: it consumes the budget and makes the
            # Trial incomplete rather than disappearing when it cannot fit.
            budget_lock = self._run_dir(run_id) / ".locks" / "execution-budget.lock"
            with self._lock(budget_lock):
                missing_control: List[Tuple[str, Any, bytes, ArtifactRef]] = []
                for item in planned_control:
                    relative_path, _value, _data, expected_ref = item
                    target = self._target(run_id, relative_path)
                    if self._exists_regular(target):
                        self._read_bytes(
                            target,
                            expected_sha256=expected_ref.sha256,
                            expected_size=expected_ref.size_bytes,
                            budget=_ReadBudget(self.max_total_read_bytes),
                            maximum_bytes=self.max_file_bytes,
                        )
                    else:
                        missing_control.append(item)

                missing_mandatory: List[Tuple[str, Any, bytes, ArtifactRef]] = []
                mandatory_bytes = 0
                for item in planned_mandatory:
                    relative_path, _value, data, expected_ref = item
                    target = self._target(run_id, relative_path)
                    if self._exists_regular(target):
                        self._read_bytes(
                            target,
                            expected_sha256=expected_ref.sha256,
                            expected_size=expected_ref.size_bytes,
                            budget=_ReadBudget(self.max_total_read_bytes),
                            maximum_bytes=execution_limit,
                        )
                    else:
                        missing_mandatory.append(item)
                        mandatory_bytes += len(data)
                missing_optional: List[Tuple[str, Any, bytes, ArtifactRef]] = []
                optional_bytes = 0
                for item in planned_optional:
                    relative_path, _value, data, expected_ref = item
                    target = self._target(run_id, relative_path)
                    if self._exists_regular(target):
                        self._read_bytes(
                            target,
                            expected_sha256=expected_ref.sha256,
                            expected_size=expected_ref.size_bytes,
                            budget=_ReadBudget(self.max_total_read_bytes),
                            maximum_bytes=execution_limit,
                        )
                    else:
                        missing_optional.append(item)
                        optional_bytes += len(data)
                if self._exists_regular(
                    self._target(run_id, submission_path)
                ):
                    raise ArtifactConflictError(
                        "submission.json already exists; use recover_trial"
                    )
                existing_trace_path = "%s/trace_ref.json" % base
                if self._exists_regular(
                    self._target(run_id, existing_trace_path)
                ):
                    raise ArtifactConflictError(
                        "trace_ref.json already exists; use recover_trial"
                    )
                if self._exists_regular(
                    self._target(run_id, receipt_path)
                ):
                    raise ArtifactConflictError(
                        "terminal receipt already exists"
                    )
                self._validate_execution_artifact_total_precommit(
                    run_id,
                    maximum_total_bytes=(
                        bundle.config.resource_budgets.max_execution_artifact_total_bytes
                    ),
                    mandatory_additional_bytes=mandatory_bytes,
                    optional_additional_bytes=optional_bytes,
                )
                for relative_path, value, _data, expected_ref in missing_control:
                    written = self._write_json(
                        run_id,
                        relative_path,
                        value,
                        maximum=self.max_file_bytes,
                    )
                    if written != expected_ref:
                        raise ArtifactIntegrityError(
                            "required Runner artifact publication changed identity"
                        )
                for relative_path, value, _data, expected_ref in missing_mandatory:
                    written = self._write_json(
                        run_id,
                        relative_path,
                        value,
                        maximum=execution_limit,
                    )
                    if written != expected_ref:
                        raise ArtifactIntegrityError(
                            "required trace publication changed its identity"
                        )
                for relative_path, value, _data, expected_ref in missing_optional:
                    written = self._write_json(
                        run_id,
                        relative_path,
                        value,
                        maximum=execution_limit,
                    )
                    if written != expected_ref:
                        raise ArtifactIntegrityError(
                            "Runner artifact publication changed its planned identity"
                        )

                written_submission = self._write_json(
                    run_id,
                    submission_path,
                    submission,
                    maximum=MAX_EVAL_SUBMISSION_BYTES,
                )
                if written_submission != submission_ref:
                    raise ArtifactIntegrityError(
                        "Submission publication changed its planned identity"
                    )
                if trace_plan is not None:
                    trace_path, trace_value, _trace_data, trace_ref = trace_plan
                    written_trace = self._write_json(
                        run_id,
                        trace_path,
                        trace_value,
                        maximum=self.max_file_bytes,
                    )
                    if written_trace != trace_ref:
                        raise ArtifactIntegrityError(
                            "trace_ref publication changed its planned identity"
                        )
                # Unique terminal receipt is always the final create-only write.
                written_receipt = self._write_json(
                    run_id,
                    receipt_path,
                    receipt,
                    maximum=MAX_STAGE_RECEIPT_BYTES,
                )
                if written_receipt != receipt_ref:
                    raise ArtifactIntegrityError(
                        "terminal receipt publication changed its planned identity"
                    )
        return self.load_trial_state(run_id, task_id, trial_id)

    def abandon_trial(
        self,
        run_id: str,
        task_id: str,
        trial_id: str,
        *,
        failure_code: FailureCode = FailureCode.PROCESS_KILLED,
        message: str = "Interrupted Trial was abandoned during recovery.",
    ) -> TrialState:
        try:
            failure_status = submission_status_for_failure(failure_code)
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                "abandonment requires a stable failed failure code"
            ) from exc
        if failure_status is not SubmissionStatus.FAILED:
            raise SchemaError("abandonment requires a stable failed failure code")
        validate_safe_text(message, "abandonment failure message")
        bundle = self._load_verified_run_bundle(
            run_id,
            budget=_ReadBudget(self.max_total_read_bytes),
        )
        plan = self._load_trial_manifest(
            bundle,
            task_id,
            trial_id,
            budget=_ReadBudget(self.max_total_read_bytes),
        )
        state, _existing_submission = self._load_trial_state(
            bundle,
            plan,
            budget=_ReadBudget(self.max_total_read_bytes),
        )
        if state.active_attempt is None:
            raise ArtifactStateError(
                "abandonment requires a started Trial attempt"
            )
        target_binding, has_prepare = self._terminal_target_binding(
            bundle,
            plan,
            attempt=state.active_attempt,
            prepare=None,
            budget=_ReadBudget(self.max_total_read_bytes),
        )
        effective_failure_code = (
            failure_code
            if has_prepare
            else FailureCode.HARNESS_MATERIALIZATION_ERROR
        )
        submission = EvalSubmission(
            schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
            task_id=task_id,
            agent_id=bundle.config.agent.agent_id,
            trial_id=trial_id,
            eval_input_digest=plan.eval_input_digest,
            target_materialization_id=target_binding,
            status=SubmissionStatus.FAILED,
            intent=None,
            review=None,
            evidence=(),
            usage=SubmissionUsage(
                elapsed_seconds=None,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                tool_calls=None,
                cost_amount=None,
                cost_currency=None,
            ),
            trace_ref=None,
            failure=SubmissionFailure(
                code=effective_failure_code,
                message=message,
                retryable=False,
            ),
        )
        # Abandonment is an ArtifactStore-owned terminal path rather than an
        # Agent execution path, but it still has to publish the same required
        # control-plane audit artifacts as a normal Runner commit.  Keep the
        # payload deliberately minimal and free of the caller's diagnostic
        # message; the canonical Submission already carries the bounded,
        # validated failure text.
        runner_artifacts = {
            "clarification_match_receipts.json": {
                "schema_version": "eval_clarification_match_receipts_v1",
                "trial_id": trial_id,
                "matcher_digest": bundle.config.clarification_matcher.digest(),
                "receipts": [],
            },
            "terminal_summary.json": {
                "schema_version": "eval_terminal_summary_v1",
                "run_id": run_id,
                "task_id": task_id,
                "trial_id": trial_id,
                "attempt": state.active_attempt,
                "status": submission.status.value,
                "failure_code": effective_failure_code.value,
                "elapsed_seconds": None,
                "stdout": {
                    "bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                    "sha256_scope": "captured_prefix",
                    "summary_bytes": 0,
                    "truncated": False,
                },
                "stderr": {
                    "bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                    "sha256_scope": "captured_prefix",
                    "summary_bytes": 0,
                    "truncated": False,
                },
                "adapter_id": "artifact-store.abandonment",
                "adapter_version": "1",
            },
        }
        return self._finalize_submission(
            run_id,
            task_id,
            trial_id,
            submission,
            attempt=state.active_attempt,
            allow_incomplete=True,
            runner_artifacts=runner_artifacts,
        )

    def load_existing_submission(
        self, run_id: str, task_id: str, trial_id: str
    ) -> EvalSubmission:
        """Read only a terminal-receipt-bound Submission; perform no repair."""

        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        plan = self._load_trial_manifest(
            bundle, task_id, trial_id, budget=budget
        )
        state, submission = self._load_trial_state(
            bundle, plan, budget=budget
        )
        if (
            state.terminal_receipt is None
            or state.status not in _TERMINAL_STATUSES
            or submission is None
        ):
            raise ArtifactStateError("nonterminal Trial has no committed Submission")
        return submission

    @staticmethod
    def _evaluation_base(
        plan: TrialManifest, evaluation_id: str
    ) -> str:
        validate_evaluation_id_shape(evaluation_id)
        return "cases/%s/trials/%s/evaluations/%s" % (
            plan.case_path_id,
            plan.trial_id,
            evaluation_id,
        )

    def _evaluation_directory_entries(self, directory: Path) -> frozenset[str]:
        if not os.path.lexists(directory):
            raise ArtifactStateError("evaluation namespace is not committed")
        self._assert_directory(directory)
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ArtifactSecurityError(
                "could not inspect evaluation namespace"
            ) from exc
        names = set()
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArtifactSecurityError(
                    "could not inspect evaluation namespace entry"
                ) from exc
            if _unsafe_node(metadata):
                raise ArtifactSecurityError(
                    "evaluation namespace contains a link or reparse point"
                )
            if entry.name == ".locks":
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ArtifactSecurityError(
                        "evaluation lock namespace is not a directory"
                    )
                self._assert_directory(Path(entry.path))
                try:
                    lock_entries = list(os.scandir(entry.path))
                except OSError as exc:
                    raise ArtifactSecurityError(
                        "could not inspect evaluation locks"
                    ) from exc
                for lock_entry in lock_entries:
                    try:
                        lock_metadata = lock_entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise ArtifactSecurityError(
                            "could not inspect evaluation lock"
                        ) from exc
                    if (
                        lock_entry.name != "evaluation.lock"
                        or _unsafe_node(lock_metadata)
                        or not stat.S_ISREG(lock_metadata.st_mode)
                    ):
                        raise ArtifactSecurityError(
                            "evaluation lock namespace contains an unsafe entry"
                        )
                names.add(entry.name)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ArtifactSecurityError(
                    "evaluation namespace contains a special file"
                )
            if entry.name not in _EVALUATION_NAMESPACE_FILENAMES:
                raise ArtifactIntegrityError(
                    "evaluation namespace contains an unknown artifact"
                )
            names.add(entry.name)
        return frozenset(names)

    def _load_evaluation_bundle(
        self,
        bundle: _VerifiedRunBundle,
        plan: TrialManifest,
        evaluation_id: str,
        *,
        budget: _ReadBudget,
    ) -> EvaluationArtifactBundle:
        state, submission = self._load_trial_state(
            bundle, plan, budget=budget
        )
        if state.status not in _TERMINAL_STATUSES or submission is None:
            raise ArtifactStateError(
                "evaluation namespace requires a terminal Submission"
            )
        base = self._evaluation_base(plan, evaluation_id)
        directory = self._target(plan.run_id, base)
        names = self._evaluation_directory_entries(directory)
        receipt_path = "%s/receipt.json" % base
        if "receipt.json" not in names:
            raise ArtifactStateError(
                "evaluation namespace has no commit receipt"
            )
        receipt = self._load_receipt(
            plan.run_id,
            receipt_path,
            budget=budget,
        )
        if (
            receipt.stage is not StageName.EVALUATOR
            or receipt.evaluation_id != evaluation_id
            or receipt.evaluation_revision is None
        ):
            raise ArtifactIntegrityError(
                "evaluation receipt has the wrong namespace binding"
            )
        self._validate_receipt_binding(
            receipt,
            plan,
            config_digest=receipt.config_digest,
        )
        refs: Dict[str, ArtifactRef] = {}
        expected_prefix = base + "/"
        for artifact in receipt.artifacts:
            filename = artifact.relative_path.rsplit("/", 1)[-1]
            if (
                artifact.relative_path != expected_prefix + filename
                or filename in refs
            ):
                raise ArtifactIntegrityError(
                    "evaluation receipt artifact path is not canonical"
                )
            refs[filename] = artifact
        required = set(_EVALUATION_JSON_ARTIFACT_NAMES)
        actual = set(refs)
        if (
            not required.issubset(actual)
            or not actual.issubset(
                required | set(_EVALUATION_OPTIONAL_ARTIFACT_NAMES)
            )
        ):
            raise ArtifactIntegrityError(
                "evaluation receipt has an incomplete or unknown artifact set"
            )
        committed_files = actual | {"receipt.json"}
        if names.difference({".locks"}) != committed_files:
            raise ArtifactIntegrityError(
                "evaluation namespace contains uncommitted or missing artifacts"
            )

        config_payload = self._read_json(
            self._target(
                plan.run_id,
                expected_prefix + "evaluator_execution_config.json",
            ),
            expected=refs["evaluator_execution_config.json"],
            budget=budget,
        )
        try:
            evaluator_execution = EvaluatorExecutionConfig.from_dict(
                config_payload
            )
        except UnsupportedProtocolVersionError:
            raise
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "evaluation execution config failed strict hydration"
            ) from exc
        execution_digest = evaluator_execution.digest()
        if execution_digest != receipt.config_digest:
            raise ArtifactIntegrityError(
                "evaluation execution config differs from receipt"
            )
        validate_evaluation_id(
            evaluation_id,
            plan.run_id,
            execution_digest,
            receipt.evaluation_revision,
        )
        evaluation_limit = self._evaluation_artifact_limit(
            evaluator_execution
        )
        if refs["evaluator_execution_config.json"].size_bytes > min(
            self.max_file_bytes, evaluation_limit
        ):
            raise ArtifactIntegrityError(
                "evaluation execution config exceeds its declared file limit"
            )

        payloads: Dict[str, str] = {}
        for filename in _EVALUATION_JSON_ARTIFACT_NAMES[1:]:
            evaluator_context_policy = (
                _EVALUATOR_CONTEXT_POLICY_BY_ARTIFACT.get(filename)
            )
            payload = self._read_json(
                self._target(plan.run_id, expected_prefix + filename),
                expected=refs[filename],
                budget=budget,
                maximum=evaluation_limit,
                evaluator_context_policy=evaluator_context_policy,
            )
            payloads[filename] = _canonical_payload_text(
                payload,
                "evaluation artifact",
                evaluator_context_policy=evaluator_context_policy,
            )
        report = None
        if "report.md" in refs:
            report = self._read_text(
                self._target(plan.run_id, expected_prefix + "report.md"),
                expected=refs["report.md"],
                budget=budget,
                maximum=evaluation_limit,
                context="evaluation report",
            )
        namespace = EvaluationNamespace(
            run_id=plan.run_id,
            task_id=plan.task_id,
            trial_id=plan.trial_id,
            evaluation_id=evaluation_id,
            evaluation_revision=receipt.evaluation_revision,
            evaluator_execution_digest=execution_digest,
            receipt=receipt,
        )
        return EvaluationArtifactBundle(
            namespace=namespace,
            evaluator_execution=evaluator_execution,
            submission_digest=submission.digest(),
            canonical_case_digest=plan.canonical_case_digest,
            trial_manifest_digest=plan.digest(),
            _intent_matches_json=payloads["intent_matches.json"],
            _review_matches_json=payloads["review_matches.json"],
            _judge_input_json=payloads["judge_input.json"],
            _judge_output_json=payloads["judge_output.json"],
            _score_json=payloads["score.json"],
            _report=report,
        )

    def load_evaluation_bundle(
        self,
        run_id: str,
        task_id: str,
        trial_id: str,
        evaluation_id: str,
    ) -> EvaluationArtifactBundle:
        """Load one receipt-committed evaluation without repairing state."""

        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        plan = self._load_trial_manifest(
            bundle, task_id, trial_id, budget=budget
        )
        return self._load_evaluation_bundle(
            bundle,
            plan,
            evaluation_id,
            budget=budget,
        )

    def load_evaluation_namespace(
        self,
        run_id: str,
        task_id: str,
        trial_id: str,
        evaluation_id: str,
    ) -> EvaluationNamespace:
        """Load only the verified metadata for one committed evaluation."""

        return self.load_evaluation_bundle(
            run_id,
            task_id,
            trial_id,
            evaluation_id,
        ).namespace

    def list_evaluations(
        self,
        run_id: str,
        task_id: Optional[str] = None,
        trial_id: Optional[str] = None,
    ) -> Tuple[EvaluationNamespace, ...]:
        """List committed Trial evaluations, failing on orphan/unsafe namespaces."""

        if (task_id is None) is not (trial_id is None):
            raise ValueError("task_id and trial_id must be provided together")
        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        if task_id is None:
            plans = tuple(
                self._load_trial_manifest(
                    bundle,
                    item.task_id,
                    item.trial_id,
                    budget=budget,
                )
                for item in bundle.manifest.trials
            )
        else:
            assert trial_id is not None
            plans = (
                self._load_trial_manifest(
                    bundle,
                    task_id,
                    trial_id,
                    budget=budget,
                ),
            )
        namespaces: List[EvaluationNamespace] = []
        for plan in plans:
            root = self._trial_dir(plan) / "evaluations"
            self._assert_directory(root)
            try:
                entries = sorted(os.scandir(root), key=lambda item: item.name)
            except OSError as exc:
                raise ArtifactSecurityError(
                    "could not inspect Trial evaluation namespaces"
                ) from exc
            for entry in entries:
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise ArtifactSecurityError(
                        "could not inspect Trial evaluation namespace"
                    ) from exc
                if _unsafe_node(metadata) or not stat.S_ISDIR(metadata.st_mode):
                    raise ArtifactSecurityError(
                        "Trial evaluations contain an unsafe entry"
                    )
                try:
                    evaluation_id = validate_evaluation_id_shape(entry.name)
                except (SchemaError, ValueError) as exc:
                    raise ArtifactIntegrityError(
                        "Trial evaluations contain an invalid namespace ID"
                    ) from exc
                loaded = self._load_evaluation_bundle(
                    bundle,
                    plan,
                    evaluation_id,
                    budget=budget,
                )
                namespaces.append(loaded.namespace)
        return tuple(
            sorted(
                namespaces,
                key=lambda item: (item.task_id, item.trial_id, item.evaluation_id),
            )
        )

    def recover_trial(
        self, run_id: str, task_id: str, trial_id: str
    ) -> ResumePlan:
        """Commit only valid orphan artifacts/receipts and mark interrupted attempts."""

        initial_budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(
            run_id, budget=initial_budget
        )
        plan = self._load_trial_manifest(
            bundle, task_id, trial_id, budget=initial_budget
        )
        with self._lock(self._trial_lock_path(plan)):
            budget = _ReadBudget(self.max_total_read_bytes)
            state, _submission = self._load_trial_state(
                bundle, plan, budget=budget
            )
            if state.terminal_receipt is not None:
                return ResumePlan(
                    trial_id=trial_id,
                    status=state.status,
                    completed_stages=state.completed_stages,
                    missing_stages=(),
                    terminal=True,
                )
            base = "cases/%s/trials/%s" % (plan.case_path_id, trial_id)
            input_path = "%s/input.json" % base
            terminal_path = self._receipt_path(plan, StageName.AGENT)
            submission_path = "%s/submission.json" % base
            prepare_committed = StageName.PREPARE in state.completed_stages
            prepare_receipt: Optional[StageReceipt] = None
            if state.active_attempt is not None:
                prepare_path = self._receipt_path(
                    plan, StageName.PREPARE, state.active_attempt
                )
                materialization_path = self._materialization_manifest_path(
                    plan, state.active_attempt
                )
            else:
                prepare_path = ""
                materialization_path = ""
            if prepare_committed:
                if state.active_attempt is None:
                    raise ArtifactIntegrityError(
                        "committed prepare receipt has no active attempt"
                    )
                prepare_receipt = self._load_receipt(
                    run_id,
                    prepare_path,
                    budget=_ReadBudget(self.max_total_read_bytes),
                )
            input_exists = self._exists_regular(
                self._target(run_id, input_path)
            )
            materialization_exists = bool(materialization_path) and (
                self._exists_regular(
                    self._target(run_id, materialization_path)
                )
            )
            terminal_exists = self._exists_regular(
                self._target(run_id, terminal_path)
            )
            submission_exists = self._exists_regular(
                self._target(run_id, submission_path)
            )
            input_ref: Optional[ArtifactRef] = None
            eval_input: Optional[EvalInput] = None
            if input_exists:
                payload, input_ref = self._adopt_json(
                    run_id,
                    input_path,
                    budget=_ReadBudget(self.max_total_read_bytes),
                    maximum=MAX_EVAL_INPUT_BYTES,
                )
                try:
                    eval_input = EvalInput.from_dict(payload)
                except UnsupportedProtocolVersionError:
                    raise
                except (SchemaError, TypeError, ValueError) as exc:
                    raise ArtifactIntegrityError(
                        "orphan EvalInput failed strict hydration"
                    ) from exc
                if eval_input.task_id != task_id:
                    raise ArtifactIntegrityError("orphan EvalInput has wrong task_id")
                if eval_input.digest() != plan.eval_input_digest:
                    raise ArtifactIntegrityError(
                        "orphan EvalInput digest does not match Trial plan"
                    )
                if state.active_attempt is None:
                    raise ArtifactIntegrityError(
                        "orphan EvalInput has no started Trial attempt"
                    )

            materialization_ref: Optional[ArtifactRef] = None
            materialization: Optional[TrialMaterializationManifest] = None
            if materialization_exists:
                materialization_payload, materialization_ref = self._adopt_json(
                    run_id,
                    materialization_path,
                    budget=_ReadBudget(self.max_total_read_bytes),
                    maximum=MAX_TRIAL_MATERIALIZATION_BYTES,
                )
                try:
                    materialization = TrialMaterializationManifest.from_dict(
                        materialization_payload
                    )
                except UnsupportedProtocolVersionError:
                    raise
                except (SchemaError, TypeError, ValueError) as exc:
                    raise ArtifactIntegrityError(
                        "orphan materialization failed strict hydration"
                    ) from exc
                if state.active_attempt is None:
                    raise ArtifactIntegrityError(
                        "orphan materialization has no started Trial attempt"
                    )
                if (
                    materialization.run_id != plan.run_id
                    or materialization.task_id != plan.task_id
                    or materialization.trial_id != plan.trial_id
                    or materialization.attempt != state.active_attempt
                    or materialization.eval_input_digest != plan.eval_input_digest
                    or materialization.wire_contract != plan.wire_contract
                    or materialization.suite_preparation_binding_digest
                    != plan.suite_preparation_binding_digest
                    or materialization.adapter_capabilities_digest
                    != plan.adapter_capabilities_digest
                ):
                    raise ArtifactIntegrityError(
                        "orphan materialization differs from immutable Trial plan"
                    )
                if eval_input is not None and (
                    materialization.review_target_digest
                    != eval_input.review_target.digest()
                ):
                    raise ArtifactIntegrityError(
                        "orphan materialization differs from EvalInput target"
                    )

            orphan_prepare: Optional[StageReceipt] = None
            if (
                not prepare_committed
                and input_ref is not None
                and eval_input is not None
                and materialization_ref is not None
                and materialization is not None
            ):
                if state.active_attempt is None:
                    raise ArtifactIntegrityError(
                        "orphan materialization has no started Trial attempt"
                    )
                orphan_prepare = StageReceipt.create(
                    run_id=run_id,
                    task_id=task_id,
                    trial_id=trial_id,
                    stage=StageName.PREPARE,
                    config_digest=plan.agent_config_digest,
                    artifacts=(input_ref, materialization_ref),
                    attempt=state.active_attempt,
                    materialization_manifest=materialization_ref,
                    materialization_manifest_digest=(
                        materialization_ref.sha256
                    ),
                    materialization_id=materialization.materialization_id,
                    eval_input_digest=materialization.eval_input_digest,
                    review_target_digest=(
                        materialization.review_target_digest
                    ),
                    prepared_source_id=materialization.prepared_source_id,
                    agent_visible_files=materialization.files,
                    adapter_capabilities_digest=(
                        materialization.adapter_capabilities_digest
                    ),
                    target_access=materialization.target_access,
                )
                self._load_prepare_materialization(
                    bundle,
                    plan,
                    orphan_prepare,
                    budget=_ReadBudget(self.max_total_read_bytes),
                )
                prepare_receipt = orphan_prepare
                prepare_committed = True

            terminal: Optional[StageReceipt] = None
            if not terminal_exists and submission_exists:
                payload, submission_ref = self._adopt_json(
                    run_id,
                    submission_path,
                    budget=_ReadBudget(self.max_total_read_bytes),
                    maximum=MAX_EVAL_SUBMISSION_BYTES,
                )
                try:
                    submission = EvalSubmission.from_dict(payload)
                except UnsupportedProtocolVersionError:
                    raise
                except (SchemaError, TypeError, ValueError) as exc:
                    raise ArtifactIntegrityError(
                        "orphan Submission failed strict hydration"
                    ) from exc
                if state.active_attempt is None:
                    raise ArtifactIntegrityError(
                        "orphan Submission has no started Trial attempt"
                    )
                self._validate_terminal_submission_binding(
                    bundle,
                    plan,
                    submission,
                    attempt=state.active_attempt,
                    prepare=prepare_receipt,
                    budget=_ReadBudget(self.max_total_read_bytes),
                )
                runner_refs = self._recoverable_runner_artifacts(
                    run_id,
                    plan,
                    attempt=state.active_attempt,
                    budget=budget,
                    maximum_bytes=min(
                        self.max_file_bytes,
                        bundle.config.resource_budgets.max_execution_artifact_file_bytes,
                    ),
                )
                runner_names = {
                    ref.relative_path.rsplit("/", 1)[-1]
                    for ref in runner_refs
                }
                missing_required = _required_runner_artifact_names(submission).difference(
                    runner_names
                )
                if missing_required:
                    raise ArtifactIntegrityError(
                        "orphan Submission lacks required Runner artifacts: %s"
                        % sorted(missing_required)
                    )
                terminal = StageReceipt.create(
                    run_id=run_id,
                    task_id=task_id,
                    trial_id=trial_id,
                    stage=StageName.AGENT,
                    config_digest=plan.agent_config_digest,
                    artifacts=(submission_ref,) + runner_refs,
                    attempt=state.active_attempt,
                    terminal_status=self._submission_status(submission),
                    failure_code=(
                        None if submission.failure is None else submission.failure.code
                    ),
                )

                self._validate_receipt_binding(
                    terminal,
                    plan,
                    config_digest=plan.agent_config_digest,
                )
                self._load_terminal_submission(
                    bundle,
                    plan,
                    terminal,
                    prepare_receipt,
                    budget=_ReadBudget(self.max_total_read_bytes),
                )

            # All candidate roots and projections are fully validated before
            # recovery publishes any missing receipt.  The run-wide execution
            # total is checked while holding the same lock as normal writers,
            # so orphan execution files cannot be blessed over budget.
            budget_lock = (
                self._run_dir(run_id) / ".locks" / "execution-budget.lock"
            )
            with self._lock(budget_lock):
                self._validate_execution_artifact_total_precommit(
                    run_id,
                    maximum_total_bytes=(
                        bundle.config.resource_budgets.max_execution_artifact_total_bytes
                    ),
                )
                if (
                    state.status is TrialStatus.RUNNING
                    and state.active_attempt is not None
                ):
                    incomplete = StageReceipt.create(
                        run_id=run_id,
                        task_id=task_id,
                        trial_id=trial_id,
                        stage=StageName.INCOMPLETE,
                        config_digest=plan.agent_config_digest,
                        attempt=state.active_attempt,
                    )
                    self._write_json(
                        run_id,
                        self._receipt_path(
                            plan, StageName.INCOMPLETE, state.active_attempt
                        ),
                        incomplete,
                        maximum=MAX_STAGE_RECEIPT_BYTES,
                    )
                    state, _submission = self._load_trial_state(
                        bundle,
                        plan,
                        budget=_ReadBudget(self.max_total_read_bytes),
                    )

                if orphan_prepare is not None:
                    self._write_json(
                        run_id,
                        prepare_path,
                        orphan_prepare,
                        maximum=MAX_STAGE_RECEIPT_BYTES,
                    )
                    prepare_committed = True

                if terminal is not None:
                    # Recovery never rewrites Submission; it only writes the missing
                    # unique terminal commit marker, and does so last.
                    self._write_json(
                        run_id,
                        terminal_path,
                        terminal,
                        maximum=MAX_STAGE_RECEIPT_BYTES,
                    )
        state = self.load_trial_state(run_id, task_id, trial_id)
        missing = () if state.status in _TERMINAL_STATUSES else tuple(
            stage
            for stage in (StageName.PREPARE, StageName.AGENT)
            if stage not in state.completed_stages
        )
        return ResumePlan(
            trial_id=trial_id,
            status=state.status,
            completed_stages=state.completed_stages,
            missing_stages=missing,
            terminal=state.status in _TERMINAL_STATUSES,
        )

    def write_evaluation(
        self,
        run_id: str,
        task_id: str,
        trial_id: str,
        *,
        evaluator_execution: EvaluatorExecutionConfig,
        revision: str,
        intent_matches: Any,
        review_matches: Any,
        judge_input: Any,
        judge_output: Any,
        score: Any,
        report: Optional[str] = None,
        resume: bool = False,
        overwrite: bool = False,
    ) -> StageReceipt:
        """Create a versioned evaluator namespace without touching Submission."""

        if type(resume) is not bool or type(overwrite) is not bool:
            raise TypeError("resume and overwrite must be bool values")
        if overwrite:
            raise ArtifactConflictError(
                "committed evaluation artifacts are immutable"
            )
        if not isinstance(evaluator_execution, EvaluatorExecutionConfig):
            raise TypeError(
                "evaluator_execution must be an EvaluatorExecutionConfig"
            )
        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        plan = self._load_trial_manifest(
            bundle, task_id, trial_id, budget=budget
        )
        state, submission = self._load_trial_state(
            bundle, plan, budget=budget
        )
        if (
            state.status not in _TERMINAL_STATUSES
            or submission is None
        ):
            raise ArtifactStateError(
                "evaluation requires a committed terminal Submission"
            )
        evaluator_execution_digest = evaluator_execution.digest()
        evaluation_limit = self._evaluation_artifact_limit(
            evaluator_execution
        )
        evaluation_id = derive_evaluation_id(
            run_id, evaluator_execution_digest, revision
        )
        base = "cases/%s/trials/%s/evaluations/%s" % (
            plan.case_path_id,
            trial_id,
            evaluation_id,
        )
        receipt_path = "%s/receipt.json" % base
        values = (
            ("evaluator_execution_config.json", evaluator_execution),
            ("intent_matches.json", intent_matches),
            ("review_matches.json", review_matches),
            ("judge_input.json", judge_input),
            ("judge_output.json", judge_output),
            ("score.json", score),
        )
        planned_json: List[Tuple[str, Any, bytes, ArtifactRef]] = []
        artifacts: List[ArtifactRef] = []
        for filename, value in values:
            relative_path = "%s/%s" % (base, filename)
            validate_safe_json(
                value,
                "evaluation artifact",
                evaluator_context_policy=_evaluator_context_policy_for_payload(
                    value,
                    _EVALUATOR_CONTEXT_POLICY_BY_ARTIFACT.get(filename),
                ),
            )
            data = canonical_json_bytes(value)
            if len(data) > min(self.max_file_bytes, evaluation_limit):
                raise ArtifactIntegrityError(
                    "evaluation artifact exceeds its execution byte limit"
                )
            artifact = self._planned_artifact_ref(
                run_id, relative_path, data
            )
            planned_json.append((relative_path, value, data, artifact))
            artifacts.append(artifact)

        planned_report: Optional[Tuple[str, str, bytes, ArtifactRef]] = None
        if report is not None:
            report_text = validate_safe_text(report, "report")
            if "\r" in report_text:
                raise SchemaError("report must use canonical LF line endings")
            report_path = "%s/report.md" % base
            report_data = report_text.encode("utf-8", "strict")
            if len(report_data) > min(self.max_file_bytes, evaluation_limit):
                raise ArtifactIntegrityError(
                    "evaluation report exceeds its execution byte limit"
                )
            report_ref = self._planned_artifact_ref(
                run_id, report_path, report_data
            )
            planned_report = (
                report_path,
                report_text,
                report_data,
                report_ref,
            )
            artifacts.append(report_ref)

        receipt = StageReceipt.create(
            run_id=run_id,
            task_id=task_id,
            trial_id=trial_id,
            stage=StageName.EVALUATOR,
            config_digest=evaluator_execution_digest,
            evaluation_id=evaluation_id,
            evaluation_revision=revision,
            artifacts=artifacts,
        )
        budget_lock = self._run_dir(run_id) / ".locks" / "execution-budget.lock"
        evaluation_lock = self._target(run_id, base) / ".locks" / "evaluation.lock"
        with self._lock(budget_lock):
            with self._lock(evaluation_lock):
                if self._exists_regular(self._target(run_id, receipt_path)):
                    existing = self._load_evaluation_bundle(
                        bundle,
                        plan,
                        evaluation_id,
                        budget=_ReadBudget(self.max_total_read_bytes),
                    )
                    if not resume:
                        raise ArtifactConflictError(
                            "evaluation version is already committed"
                        )
                    expected_values = (
                        intent_matches,
                        review_matches,
                        judge_input,
                        judge_output,
                        score,
                    )
                    existing_values = (
                        existing.intent_matches,
                        existing.review_matches,
                        existing.judge_input,
                        existing.judge_output,
                        existing.score,
                    )
                    if (
                        existing.evaluator_execution != evaluator_execution
                        or existing.report != report
                        or any(
                            canonical_json_bytes(actual)
                            != canonical_json_bytes(expected)
                            for actual, expected in zip(
                                existing_values, expected_values
                            )
                        )
                    ):
                        raise ArtifactConflictError(
                            "committed evaluation differs from resume inputs"
                        )
                    return existing.receipt
                existing_names = self._evaluation_directory_entries(
                    self._target(run_id, base)
                )
                planned_names = {
                    relative_path.rsplit("/", 1)[-1]
                    for relative_path, _value, _data, _artifact in planned_json
                }
                if planned_report is not None:
                    planned_names.add("report.md")
                if existing_names.difference({".locks"}).difference(
                    planned_names
                ):
                    raise ArtifactIntegrityError(
                        "evaluation namespace contains an unplanned orphan artifact"
                    )
                missing_json: List[Tuple[str, Any, bytes, ArtifactRef]] = []
                missing_bytes = 0
                for item in planned_json:
                    relative_path, _value, data, artifact = item
                    target = self._target(run_id, relative_path)
                    if self._exists_regular(target):
                        self._read_bytes(
                            target,
                            expected_sha256=artifact.sha256,
                            expected_size=artifact.size_bytes,
                            budget=_ReadBudget(self.max_total_read_bytes),
                        )
                    else:
                        missing_json.append(item)
                        missing_bytes += len(data)

                report_missing = False
                if planned_report is not None:
                    report_path, _report_text, report_data, report_ref = planned_report
                    target = self._target(run_id, report_path)
                    if self._exists_regular(target):
                        self._read_bytes(
                            target,
                            expected_sha256=report_ref.sha256,
                            expected_size=report_ref.size_bytes,
                            budget=_ReadBudget(self.max_total_read_bytes),
                        )
                    else:
                        report_missing = True
                        missing_bytes += len(report_data)

                current_bytes = self._execution_artifact_total_bytes(run_id)
                maximum_total = (
                    evaluator_execution.max_execution_artifact_total_bytes
                )
                if current_bytes + missing_bytes > maximum_total:
                    raise ArtifactIntegrityError(
                        "execution artifacts exceed the configured cumulative byte limit"
                    )

                for relative_path, value, _data, expected_ref in missing_json:
                    written = self._write_json(
                        run_id,
                        relative_path,
                        value,
                        maximum=evaluation_limit,
                        evaluator_context_policy=(
                            _EVALUATOR_CONTEXT_POLICY_BY_ARTIFACT.get(
                                relative_path.rsplit("/", 1)[-1]
                            )
                        ),
                    )
                    if written != expected_ref:
                        raise ArtifactIntegrityError(
                            "evaluation artifact publication changed its planned identity"
                        )
                if report_missing and planned_report is not None:
                    report_path, report_text, _report_data, expected_ref = planned_report
                    written = self._write_text(
                        run_id,
                        report_path,
                        report_text,
                        maximum=evaluation_limit,
                    )
                    if written != expected_ref:
                        raise ArtifactIntegrityError(
                            "evaluation report publication changed its planned identity"
                        )
                # Evaluator receipt is the version commit marker and is last.
                self._write_json(
                    run_id,
                    receipt_path,
                    receipt,
                    maximum=MAX_STAGE_RECEIPT_BYTES,
                )
        return receipt

    @staticmethod
    def _validate_run_summary_binding(
        bundle: _VerifiedRunBundle,
        payload: Any,
        *,
        evaluation_id: str,
        evaluator_execution_digest: str,
        evaluation_revision: str,
    ) -> str:
        if type(payload) is not dict:
            raise ArtifactIntegrityError(
                "Run evaluation summary must be a JSON object"
            )
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
            raise ArtifactIntegrityError(
                "Run evaluation summary has an invalid top-level schema"
            )
        # Import lazily: report.py intentionally imports ArtifactStore models.
        from .report import REPORT_REVISION, RUN_REPORT_SUMMARY_SCHEMA_VERSION

        if (
            payload["schema_version"] != RUN_REPORT_SUMMARY_SCHEMA_VERSION
            or payload["report_revision"] != REPORT_REVISION
        ):
            raise ArtifactIntegrityError(
                "Run evaluation summary has an unsupported schema or revision"
            )
        source_bindings = payload["source_bindings"]
        if type(source_bindings) is not dict:
            raise ArtifactIntegrityError(
                "Run evaluation summary source bindings are invalid"
            )
        expected_bindings = {
            "run_id": bundle.config.run_id,
            "run_config_digest": bundle.config.digest(),
            "run_manifest_digest": bundle.manifest.digest(),
            "case_snapshot_id": bundle.config.suite.case_snapshot_id,
            "case_snapshot_digest": bundle.config.suite.case_snapshot_digest,
            "evaluation_id": evaluation_id,
            "evaluation_revision": evaluation_revision,
            "evaluator_execution_digest": evaluator_execution_digest,
        }
        if any(
            source_bindings.get(name) != value
            for name, value in expected_bindings.items()
        ):
            raise ArtifactIntegrityError(
                "Run evaluation summary differs from verified Run sources"
            )
        identity = payload["identity"]
        if type(identity) is not dict:
            raise ArtifactIntegrityError(
                "Run evaluation summary identity is invalid"
            )
        expected_suite = {
            "suite_id": bundle.config.suite.suite_id,
            "suite_version": bundle.config.suite.suite_version,
            "manifest_digest": bundle.config.suite.manifest_digest,
            "case_snapshot_id": bundle.config.suite.case_snapshot_id,
            "case_snapshot_digest": bundle.config.suite.case_snapshot_digest,
        }
        evaluator_identity = identity.get("evaluator")
        if (
            identity.get("suite") != expected_suite
            or identity.get("agent") != bundle.config.agent.to_dict()
            or type(evaluator_identity) is not dict
            or evaluator_identity.get("execution_config_digest")
            != evaluator_execution_digest
            or evaluator_identity.get("evaluation_id") != evaluation_id
            or evaluator_identity.get("evaluation_revision")
            != evaluation_revision
        ):
            raise ArtifactIntegrityError(
                "Run evaluation summary identity differs from verified sources"
            )
        summary_id = payload["summary_id"]
        validate_path_segment(summary_id, "Run evaluation summary_id")
        identity_payload = dict(payload)
        identity_payload.pop("summary_id")
        if summary_id != stable_id(
            "run-report-summary-v1", identity_payload
        ):
            raise ArtifactIntegrityError(
                "Run evaluation summary ID is not canonical"
            )
        return summary_id

    def _run_evaluation_directory_entries(
        self, directory: Path
    ) -> frozenset[str]:
        if not os.path.lexists(directory):
            return frozenset()
        self._assert_directory(directory)
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ArtifactSecurityError(
                "could not inspect Run evaluation namespace"
            ) from exc
        names = set()
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArtifactSecurityError(
                    "could not inspect Run evaluation artifact"
                ) from exc
            if _unsafe_node(metadata) or not stat.S_ISREG(metadata.st_mode):
                raise ArtifactSecurityError(
                    "Run evaluation namespace contains an unsafe entry"
                )
            if entry.name not in _RUN_EVALUATION_FILENAMES:
                raise ArtifactIntegrityError(
                    "Run evaluation namespace contains an unknown artifact"
                )
            names.add(entry.name)
        return frozenset(names)

    def load_run_evaluation(
        self, run_id: str, evaluation_id: str
    ) -> RunEvaluationBundle:
        """Load one committed Run summary/report pair without repair."""

        validate_evaluation_id_shape(evaluation_id)
        budget = _ReadBudget(self.max_total_read_bytes)
        bundle = self._load_verified_run_bundle(run_id, budget=budget)
        base = "evaluations/%s" % evaluation_id
        directory = self._target(run_id, base)
        names = self._run_evaluation_directory_entries(directory)
        if not names:
            raise ArtifactStateError("Run evaluation namespace is not committed")
        if names != _RUN_EVALUATION_FILENAMES:
            raise ArtifactIntegrityError(
                "Run evaluation namespace is incomplete"
            )
        summary_path = self._target(run_id, base + "/summary.json")
        summary_payload = self._read_json(
            summary_path,
            budget=budget,
        )
        if type(summary_payload) is not dict:
            raise ArtifactIntegrityError(
                "Run evaluation summary must be an object"
            )
        source_bindings = summary_payload.get("source_bindings")
        if type(source_bindings) is not dict:
            raise ArtifactIntegrityError(
                "Run evaluation summary source bindings are invalid"
            )
        try:
            evaluator_execution_digest = _digest(
                source_bindings.get("evaluator_execution_digest"),
                "Run evaluation evaluator_execution_digest",
            )
            evaluation_revision = validate_path_segment(
                source_bindings.get("evaluation_revision"),
                "Run evaluation revision",
            )
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "Run evaluation source identity is invalid"
            ) from exc
        summary_id = self._validate_run_summary_binding(
            bundle,
            summary_payload,
            evaluation_id=evaluation_id,
            evaluator_execution_digest=evaluator_execution_digest,
            evaluation_revision=evaluation_revision,
        )
        report_path = self._target(run_id, base + "/report.md")
        report = self._read_text(
            report_path,
            budget=budget,
            context="Run evaluation report",
            allow_rendered_environment_projection=True,
        )
        summary_data = canonical_json_bytes(summary_payload)
        report_data = report.encode("utf-8", "strict")
        namespace = RunEvaluationNamespace(
            schema_version=EVAL_RUN_EVALUATION_NAMESPACE_SCHEMA_VERSION,
            run_id=run_id,
            evaluation_id=evaluation_id,
            evaluation_revision=evaluation_revision,
            evaluator_execution_digest=evaluator_execution_digest,
            summary_id=summary_id,
            summary=self._artifact_ref(run_id, summary_path, summary_data),
            report=self._artifact_ref(run_id, report_path, report_data),
        )
        return RunEvaluationBundle(
            namespace=namespace,
            _summary_json=summary_data.decode("utf-8", "strict"),
            _report=report,
        )

    def list_run_evaluations(
        self, run_id: str
    ) -> Tuple[RunEvaluationNamespace, ...]:
        """List committed Run report namespaces in stable ID order."""

        bundle = self._load_verified_run_bundle(run_id)
        root = self._run_dir(bundle.config.run_id) / "evaluations"
        self._assert_directory(root)
        try:
            entries = sorted(os.scandir(root), key=lambda item: item.name)
        except OSError as exc:
            raise ArtifactSecurityError(
                "could not inspect Run evaluation namespaces"
            ) from exc
        values: List[RunEvaluationNamespace] = []
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArtifactSecurityError(
                    "could not inspect Run evaluation namespace"
                ) from exc
            if _unsafe_node(metadata) or not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactSecurityError(
                    "Run evaluations contain an unsafe entry"
                )
            try:
                evaluation_id = validate_evaluation_id_shape(entry.name)
            except (SchemaError, ValueError) as exc:
                raise ArtifactIntegrityError(
                    "Run evaluations contain an invalid namespace ID"
                ) from exc
            values.append(
                self.load_run_evaluation(run_id, evaluation_id).namespace
            )
        return tuple(sorted(values, key=lambda item: item.evaluation_id))

    def write_run_evaluation(
        self,
        run_id: str,
        *,
        evaluator_execution: EvaluatorExecutionConfig,
        revision: str,
        summary: Any,
        report: Optional[str] = None,
        resume: bool = False,
        overwrite: bool = False,
    ) -> RunEvaluationBundle:
        """Create one immutable Run summary/report pair, summary commit marker last."""

        if type(resume) is not bool or type(overwrite) is not bool:
            raise TypeError("resume and overwrite must be bool values")
        if overwrite:
            raise ArtifactConflictError(
                "Run evaluation artifacts are immutable and cannot be overwritten"
            )
        if type(evaluator_execution) is not EvaluatorExecutionConfig:
            raise TypeError(
                "evaluator_execution must be an EvaluatorExecutionConfig"
            )
        revision = validate_path_segment(revision, "evaluation revision")
        from .report import RunReportSummary, render_run_markdown

        if type(summary) is not RunReportSummary:
            raise TypeError("summary must be a sealed RunReportSummary")
        bundle = self._load_verified_run_bundle(run_id)
        execution_digest = evaluator_execution.digest()
        evaluation_id = derive_evaluation_id(
            run_id, execution_digest, revision
        )
        summary_payload = summary.to_dict()
        self._validate_run_summary_binding(
            bundle,
            summary_payload,
            evaluation_id=evaluation_id,
            evaluator_execution_digest=execution_digest,
            evaluation_revision=revision,
        )
        rendered = render_run_markdown(summary)
        if report is None:
            report_text = rendered
        else:
            report_text = _validated_artifact_text(
                report,
                "Run evaluation report",
                allow_rendered_environment_projection=True,
            )
            if report_text != rendered:
                raise ArtifactIntegrityError(
                    "Run evaluation report differs from pure summary rendering"
                )
        if "\r" in report_text:
            raise SchemaError(
                "Run evaluation report must use canonical LF line endings"
            )
        summary_data = canonical_json_bytes(summary_payload)
        report_data = report_text.encode("utf-8", "strict")
        evaluation_limit = self._evaluation_artifact_limit(
            evaluator_execution
        )
        effective_limit = min(self.max_file_bytes, evaluation_limit)
        if len(summary_data) > effective_limit or len(report_data) > effective_limit:
            raise ArtifactIntegrityError(
                "Run evaluation artifact exceeds its execution byte limit"
            )
        base = "evaluations/%s" % evaluation_id
        summary_path = base + "/summary.json"
        report_path = base + "/report.md"
        summary_ref = self._planned_artifact_ref(
            run_id, summary_path, summary_data
        )
        report_ref = self._planned_artifact_ref(
            run_id, report_path, report_data
        )
        run_lock = self._run_dir(run_id) / ".locks" / (
            "run-evaluation-%s.lock" % evaluation_id
        )
        budget_lock = self._run_dir(run_id) / ".locks" / "execution-budget.lock"
        with self._lock(budget_lock):
            with self._lock(run_lock):
                directory = self._target(run_id, base)
                names = self._run_evaluation_directory_entries(directory)
                if names and not resume:
                    raise ArtifactConflictError(
                        "Run evaluation namespace already exists; use resume"
                    )
                if names.difference(_RUN_EVALUATION_FILENAMES):
                    raise ArtifactIntegrityError(
                        "Run evaluation namespace contains unknown artifacts"
                    )
                has_summary = "summary.json" in names
                has_report = "report.md" in names
                if has_summary:
                    if not has_report:
                        raise ArtifactIntegrityError(
                            "committed Run summary is missing report.md"
                        )
                    try:
                        self._read_bytes(
                            self._target(run_id, summary_path),
                            expected_sha256=summary_ref.sha256,
                            expected_size=summary_ref.size_bytes,
                            budget=_ReadBudget(self.max_total_read_bytes),
                            maximum_bytes=evaluation_limit,
                        )
                        self._read_bytes(
                            self._target(run_id, report_path),
                            expected_sha256=report_ref.sha256,
                            expected_size=report_ref.size_bytes,
                            budget=_ReadBudget(self.max_total_read_bytes),
                            maximum_bytes=evaluation_limit,
                        )
                    except ArtifactIntegrityError as exc:
                        raise ArtifactConflictError(
                            "existing Run evaluation differs from requested sources"
                        ) from exc
                    return self.load_run_evaluation(run_id, evaluation_id)

                missing_bytes = len(summary_data)
                if has_report:
                    try:
                        self._read_bytes(
                            self._target(run_id, report_path),
                            expected_sha256=report_ref.sha256,
                            expected_size=report_ref.size_bytes,
                            budget=_ReadBudget(self.max_total_read_bytes),
                            maximum_bytes=evaluation_limit,
                        )
                    except ArtifactIntegrityError as exc:
                        raise ArtifactConflictError(
                            "orphan Run report differs from requested sources"
                        ) from exc
                else:
                    missing_bytes += len(report_data)
                current_bytes = self._execution_artifact_total_bytes(run_id)
                if (
                    current_bytes + missing_bytes
                    > evaluator_execution.max_execution_artifact_total_bytes
                ):
                    raise ArtifactIntegrityError(
                        "execution artifacts exceed the configured cumulative byte limit"
                    )
                if not has_report:
                    written_report = self._write_text(
                        run_id,
                        report_path,
                        report_text,
                        maximum=evaluation_limit,
                        allow_rendered_environment_projection=True,
                    )
                    if written_report != report_ref:
                        raise ArtifactIntegrityError(
                            "Run report publication changed its planned identity"
                        )
                # summary.json is the authoritative projection and commit
                # marker.  Resume may complete an exact orphan report, but
                # nothing writes after a committed summary.
                written_summary = self._write_json(
                    run_id,
                    summary_path,
                    summary_payload,
                    maximum=evaluation_limit,
                )
                if written_summary != summary_ref:
                    raise ArtifactIntegrityError(
                        "Run summary publication changed its planned identity"
                    )
        return self.load_run_evaluation(run_id, evaluation_id)


def load_existing_submission(
    runs_root: os.PathLike[str] | str,
    run_id: str,
    task_id: str,
    trial_id: str,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_read_bytes: int = DEFAULT_MAX_TOTAL_READ_BYTES,
) -> EvalSubmission:
    """Read-only convenience loader for evaluator-only processes."""

    return ArtifactStore(
        runs_root,
        max_file_bytes=max_file_bytes,
        max_total_read_bytes=max_total_read_bytes,
        create_root=False,
    ).load_existing_submission(run_id, task_id, trial_id)


__all__ = [
    "EVAL_RUN_MANIFEST_SCHEMA_VERSION",
    "EVAL_TRIAL_MANIFEST_SCHEMA_VERSION",
    "EVAL_STAGE_RECEIPT_SCHEMA_VERSION",
    "EVAL_RUN_PREFLIGHT_SCHEMA_VERSION",
    "EVAL_PREFLIGHT_CANDIDATE_SCHEMA_VERSION",
    "EVAL_TRIAL_MATERIALIZATION_SCHEMA_VERSION",
    "PRE_MATERIALIZATION_FAILURE_BINDING_VERSION",
    "EVAL_RUN_EVALUATION_NAMESPACE_SCHEMA_VERSION",
    "MAX_RUN_MANIFEST_BYTES",
    "MAX_TRIAL_MANIFEST_BYTES",
    "MAX_STAGE_RECEIPT_BYTES",
    "MAX_TRIAL_MATERIALIZATION_BYTES",
    "MAX_RUN_PREFLIGHT_BYTES",
    "MAX_PREFLIGHT_CANDIDATE_BYTES",
    "MAX_RUNNER_ARTIFACTS",
    "MAX_RUNNER_ARTIFACT_NAME_CHARS",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_TOTAL_READ_BYTES",
    "DIRECTORY_FSYNC_SUPPORTED",
    "ArtifactError",
    "ArtifactConflictError",
    "ArtifactIntegrityError",
    "ExecutionArtifactBudgetError",
    "RequiredExecutionArtifactBudgetError",
    "ArtifactSecurityError",
    "ArtifactStateError",
    "RunStatus",
    "StageName",
    "ArtifactRef",
    "TargetAccess",
    "AgentVisibleFileBinding",
    "TrialMaterializationManifest",
    "TrialManifest",
    "RunTrialPlan",
    "RunManifest",
    "StageReceipt",
    "TrialState",
    "VerifiedTrialMaterialization",
    "RunState",
    "ResumePlan",
    "EvaluationNamespace",
    "EvaluationArtifactBundle",
    "RunEvaluationNamespace",
    "RunEvaluationBundle",
    "derive_receipt_id",
    "derive_pre_materialization_failure_binding",
    "ArtifactStore",
    "load_existing_submission",
]
