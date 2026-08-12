"""Black-box adapter for this repository's formal ``review-agent`` CLI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from review_agent.diff_artifact import DiffArtifact, DiffArtifactStore
from review_agent.execution_profile import AgentExecutionProfile
from review_agent.pr_workspace import PRWorkspaceStore, SessionWorkspace
from review_agent.review_protocol import (
    FindingSeverity as ProductFindingSeverityV2,
    IntentSource as ProductIntentSourceV2,
    IntentVersionEnvelope,
    ReviewResult as ProductReviewResultV2,
)
from review_agent.revision import RevisionResolver, canonical_repository_identity
from review_agent.run_state import RunStatus
from review_agent.session import PhaseStatus
from review_agent.session_store import SessionV6Store

from ..clarification import ClarificationChannel
from ..config import AdapterCapabilitiesV2
from ..artifacts import TargetAccess
from ..models import (
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    MAX_EVAL_INPUT_BYTES,
    EvalInput,
    EvalSubmission,
    DiffSide,
    EvidenceKind,
    FailureCode,
    FindingSeverity,
    IntentClaimSource,
    IntentDimension,
    IntentResult,
    MAX_EVIDENCE_EXCERPT_BYTES,
    ReviewTargetKind,
    RepositoryDiffEvidenceSource,
    RepositoryReviewTarget,
    SubmissionEvidence,
    SubmissionFinding,
    SubmissionIntent,
    SubmissionIntentClaim,
    SubmissionReview,
    SubmissionStatus,
    TraceRef,
    TraceType,
    canonical_json_bytes,
    stable_id,
)
from ..repository import repository_from_eval_input
from ..submission import (
    empty_usage,
    failure_submission,
    validate_submission_trace,
)
from .base import (
    AdapterCompatibility,
    AdapterIncompatibilityReason,
    AgentAdapterError,
    AgentAdapterIncompatibleError,
    AgentRunConfig,
)
from .subprocess_agent import (
    BoundedProcessResult,
    build_subprocess_environment,
    returncode_was_killed,
    run_bounded_process,
)


CURRENT_AGENT_ADAPTER_KIND = "current-agent-cli-v2"
CURRENT_AGENT_ADAPTER_VERSION = "3"


def current_agent_capabilities() -> AdapterCapabilitiesV2:
    """Return the fixed capability declaration for the product CLI adapter."""

    return AdapterCapabilitiesV2.from_dict(
        {
            "schema_version": "eval_adapter_capabilities_v2",
            "adapter_id": CURRENT_AGENT_ADAPTER_KIND,
            "adapter_version": CURRENT_AGENT_ADAPTER_VERSION,
            "input_schema_version": EVAL_INPUT_SCHEMA_VERSION,
            "submission_schema_version": EVAL_SUBMISSION_SCHEMA_VERSION,
            "target_kinds": [ReviewTargetKind.REPOSITORY.value],
            "evidence_kinds": [
                EvidenceKind.REPOSITORY_FILE.value,
                EvidenceKind.REPOSITORY_DIFF.value,
                EvidenceKind.COMMAND_OUTPUT.value,
                EvidenceKind.EXTERNAL_RECORD.value,
            ],
            "clarification_protocol": "canonical-clarification-v2",
            "trace_protocol": "local-trace-v2",
            "subprocess_wire_version": None,
            "isolation_profile": "repository-worktree-v2",
        }
    )

_ADAPTER_FIELDS = frozenset(
    {
        "kind",
        "command",
        "review_arguments",
        "environment_allowlist",
    }
)
_ENVIRONMENT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MAX_ARGUMENTS = 256
_MAX_ARGUMENT_CHARS = 8_192
_MAX_ENVIRONMENT_KEYS = 128
_MAX_PRODUCT_JSON_BYTES = 16 * 1024 * 1024
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_CI_EVIDENCE_BUNDLE_SCHEMA_VERSION = "review_agent_ci_evidence_bundle_v1"
_CI_EVIDENCE_CONTROL_DIRECTORY = PurePosixPath(".review-agent/eval-input")
_CI_EVIDENCE_BUNDLE_PREFIX = "existing-ci-evidence."
_CI_EVIDENCE_BUNDLE_SUFFIX = ".v1.json"

_FORBIDDEN_REVIEW_ARGUMENTS = frozenset(
    {
        "--repo",
        "--base",
        "--head",
        "--intent",
        "--focus",
        "--title",
        "--description",
        "--requirement",
        "--project-rule",
        "--ci-evidence",
        "--ci-evidence-file",
        "--external-review-id",
        "--workspace-root",
        "--format",
        "--reviewer-mode",
        "--reviewer-loop",
        "--memory-mode",
        "--memory-root",
        "--non-interactive",
    }
)


class _CurrentAdapterError(RuntimeError):
    pass


class _CurrentArtifactError(_CurrentAdapterError):
    pass


def _storage_path(path: Path) -> Path:
    """Use the extended-length namespace for Windows filesystem syscalls."""

    raw = os.fspath(path)
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return Path(raw)
    absolute = os.path.abspath(raw)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


@dataclass(frozen=True)
class _CurrentAdapterConfiguration:
    command: Tuple[str, ...]
    review_arguments: Tuple[str, ...]
    environment_allowlist: Tuple[str, ...]
    agent_snapshot_digest: str
    execution_profile: AgentExecutionProfile
    execution_profile_digest: str


def _argument_array(
    value: Any,
    context: str,
    *,
    require_absolute: bool,
    allow_empty: bool = False,
) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _CurrentAdapterError("%s must be an array" % context)
    if (not value and not allow_empty) or len(value) > _MAX_ARGUMENTS:
        raise _CurrentAdapterError("%s has an invalid size" % context)
    result = []
    for index, item in enumerate(value):
        if type(item) is not str or not item or len(item) > _MAX_ARGUMENT_CHARS:
            raise _CurrentAdapterError("%s contains an invalid argument" % context)
        if "\x00" in item:
            raise _CurrentAdapterError("%s contains a null byte" % context)
        if index == 0 and require_absolute and not Path(item).is_absolute():
            raise _CurrentAdapterError("current Agent executable must be absolute")
        result.append(item)
    return tuple(result)


def _configuration(config: AgentRunConfig) -> _CurrentAdapterConfiguration:
    snapshot = config.agent
    digest = snapshot.digest()
    raw = snapshot.parameters.get("adapter")
    if not isinstance(raw, Mapping) or set(raw) != _ADAPTER_FIELDS:
        raise _CurrentAdapterError("current Agent adapter configuration is invalid")
    if raw["kind"] != CURRENT_AGENT_ADAPTER_KIND:
        raise _CurrentAdapterError("current Agent adapter kind is unsupported")
    command = _argument_array(raw["command"], "adapter.command", require_absolute=True)
    review_arguments = _argument_array(
        raw["review_arguments"],
        "adapter.review_arguments",
        require_absolute=False,
        allow_empty=True,
    )
    for argument in review_arguments:
        option = argument.split("=", 1)[0]
        ci_file_abbreviation = (
            option.startswith("--")
            and len(option) > 2
            and "--ci-evidence-file".startswith(option)
        )
        if option in _FORBIDDEN_REVIEW_ARGUMENTS or ci_file_abbreviation:
            raise _CurrentAdapterError(
                "current Agent fixed arguments override invocation authority"
            )
    environment_raw = raw["environment_allowlist"]
    if isinstance(environment_raw, (str, bytes)) or not isinstance(
        environment_raw, Sequence
    ) or len(environment_raw) > _MAX_ENVIRONMENT_KEYS:
        raise _CurrentAdapterError("adapter environment allowlist is invalid")
    environment = []
    seen = set()
    for item in environment_raw:
        if type(item) is not str or _ENVIRONMENT_KEY_RE.fullmatch(item) is None:
            raise _CurrentAdapterError("adapter environment key is invalid")
        folded = item.casefold()
        if folded in seen:
            raise _CurrentAdapterError("adapter environment key is duplicated")
        seen.add(folded)
        environment.append(item)
    if snapshot.digest() != digest:
        raise _CurrentAdapterError("current Agent snapshot changed during validation")
    profile_binding = snapshot.parameters.get("agent_execution_profile")
    if not isinstance(profile_binding, Mapping) or set(profile_binding) != {
        "profile",
        "digest",
    }:
        raise _CurrentAdapterError(
            "current Agent execution profile binding is invalid"
        )
    try:
        execution_profile = AgentExecutionProfile.from_dict(
            profile_binding["profile"]
        )
    except (TypeError, ValueError) as exc:
        raise _CurrentAdapterError(
            "current Agent execution profile is invalid"
        ) from exc
    execution_profile_digest = profile_binding["digest"]
    if (
        type(execution_profile_digest) is not str
        or execution_profile.digest() != execution_profile_digest
    ):
        raise _CurrentAdapterError(
            "current Agent execution profile digest is invalid"
        )
    from review_agent.command import review_execution_profile_from_arguments

    expected_profile = review_execution_profile_from_arguments(review_arguments)
    if (
        expected_profile.digest() != execution_profile.digest()
        or expected_profile.to_dict() != execution_profile.to_dict()
    ):
        raise AgentAdapterIncompatibleError(
            AdapterIncompatibilityReason.EXECUTION_PROFILE_MISMATCH
        )
    return _CurrentAdapterConfiguration(
        command=command,
        review_arguments=review_arguments,
        environment_allowlist=tuple(environment),
        agent_snapshot_digest=digest,
        execution_profile=execution_profile,
        execution_profile_digest=execution_profile_digest,
    )


def _failure(
    *,
    eval_input: EvalInput,
    config: AgentRunConfig,
    target_materialization_id: str,
    code: FailureCode,
    elapsed: float,
    retryable: bool,
    intent: Optional[SubmissionIntent] = None,
    trace_ref: Optional[TraceRef] = None,
    workspace: Optional[Path] = None,
) -> EvalSubmission:
    messages = {
        FailureCode.TIMEOUT: "Current Agent process exceeded its time limit",
        FailureCode.NON_ZERO_EXIT: "Current Agent process exited unsuccessfully",
        FailureCode.PROCESS_KILLED: "Current Agent process was killed",
        FailureCode.OUTPUT_OVERFLOW: "Current Agent output exceeded its byte limit",
        FailureCode.SCHEMA_MISMATCH: "Current Agent artifacts failed validation",
        FailureCode.CLARIFICATION_REQUIRED: "Current Agent requires clarification",
        FailureCode.ADAPTER_ERROR: "Current Agent adapter boundary failed",
        FailureCode.UNKNOWN: "Current Agent ended without a valid terminal state",
    }
    submission = failure_submission(
        eval_input=eval_input,
        config=config,
        target_materialization_id=target_materialization_id,
        code=code,
        message=messages[code],
        retryable=retryable,
        intent=intent,
        usage=empty_usage(elapsed_seconds=max(0.0, elapsed)),
        trace_ref=trace_ref,
    )
    if trace_ref is not None and workspace is not None:
        try:
            return validate_submission_trace(
                submission,
                workspace=workspace,
                max_trace_bytes=config.max_trace_bytes,
            )
        except AgentAdapterError:
            return replace(submission, trace_ref=None)
    return submission


def _read_bounded_regular_file(path: Path, maximum: int, context: str) -> bytes:
    try:
        storage_path = _storage_path(path)
        before = os.lstat(storage_path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or getattr(before, "st_file_attributes", 0) & _REPARSE_POINT
            or before.st_nlink != 1
            or before.st_size > maximum
        ):
            raise _CurrentArtifactError("%s is not a bounded regular file" % context)
        with storage_path.open("rb") as handle:
            data = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
        if (
            len(data) > maximum
            or after.st_size != len(data)
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise _CurrentArtifactError("%s changed while it was read" % context)
        return data
    except _CurrentArtifactError:
        raise
    except OSError as exc:
        raise _CurrentArtifactError("%s could not be read safely" % context) from exc


def _strict_json_object(data: bytes, context: str) -> Dict[str, Any]:
    if len(data) > _MAX_PRODUCT_JSON_BYTES:
        raise _CurrentArtifactError("%s exceeds its byte limit" % context)

    def reject_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _CurrentArtifactError("%s contains duplicate keys" % context)
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8", "strict"),
            object_pairs_hook=reject_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                _CurrentArtifactError("%s contains non-finite numbers" % context)
            ),
        )
    except _CurrentArtifactError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise _CurrentArtifactError("%s is invalid JSON" % context) from exc
    if type(value) is not dict:
        raise _CurrentArtifactError("%s must contain an object" % context)
    return value


def _load_registered_json(
    *,
    run_dir: Path,
    store: SessionStore,
    manifest: SessionManifest,
    name: str,
    expected_path: str,
    expected_phase: RunPhase,
    expected_revision: str,
) -> Dict[str, Any]:
    descriptor = manifest.artifacts.get(name)
    if (
        descriptor is None
        or descriptor.path != expected_path
        or descriptor.schema != artifact_schema(name)
        or descriptor.phase is not expected_phase
        or descriptor.revision_binding != expected_revision
        or not store.validate_artifact(descriptor)
    ):
        raise _CurrentArtifactError("current Agent artifact descriptor is invalid")
    path = run_dir.joinpath(*PurePosixPath(descriptor.path).parts)
    data = _read_bounded_regular_file(path, _MAX_PRODUCT_JSON_BYTES, name)
    if hashlib.sha256(data).hexdigest() != descriptor.sha256:
        raise _CurrentArtifactError("current Agent artifact changed after validation")
    return _strict_json_object(data, name)


@dataclass(frozen=True)
class _CurrentV6Run:
    runtime_root: Path
    workspace_store: PRWorkspaceStore
    session: SessionWorkspace
    review_result: ProductReviewResultV2
    diff: DiffArtifact
    intent: IntentVersionEnvelope


def _single_managed_directory(
    parent: Path,
    pattern: re.Pattern[str],
    context: str,
) -> Path:
    try:
        parent_info = os.lstat(parent)
        if (
            not stat.S_ISDIR(parent_info.st_mode)
            or stat.S_ISLNK(parent_info.st_mode)
            or getattr(parent_info, "st_file_attributes", 0) & _REPARSE_POINT
        ):
            raise _CurrentArtifactError(f"{context} root is unsafe")
        values: list[Path] = []
        with os.scandir(parent) as entries:
            for entry in entries:
                info = os.lstat(entry.path)
                if (
                    not stat.S_ISDIR(info.st_mode)
                    or stat.S_ISLNK(info.st_mode)
                    or getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
                    or pattern.fullmatch(entry.name) is None
                ):
                    raise _CurrentArtifactError(
                        f"{context} contains an unsafe entry"
                    )
                values.append(Path(entry.path))
    except _CurrentArtifactError:
        raise
    except OSError as error:
        raise _CurrentArtifactError(f"{context} is unavailable") from error
    if len(values) != 1:
        raise _CurrentArtifactError(
            f"{context} must contain exactly one directory"
        )
    return values[0]


def _discover_v6_run(
    *,
    runtime_root: Path,
    workspace: Path,
    eval_input: EvalInput,
    stdout: bytes,
) -> _CurrentV6Run:
    product_workspace = _single_managed_directory(
        runtime_root / "pr",
        re.compile(r"p-[0-9a-f]{32}"),
        "current Agent PRWorkspace catalog",
    )
    snapshot_path = _single_managed_directory(
        product_workspace / "Snapshots",
        re.compile(r"s-[0-9a-f]{32}"),
        "current Agent Snapshot catalog",
    )
    session_path = _single_managed_directory(
        product_workspace / "Sessions",
        re.compile(r"u-[0-9a-f]{32}"),
        "current Agent Session catalog",
    )
    pr_payload = _strict_json_object(
        _read_bounded_regular_file(
            product_workspace / "PR" / "pr.json",
            _MAX_PRODUCT_JSON_BYTES,
            "current Agent PR metadata",
        ),
        "current Agent PR metadata",
    )
    snapshot_payload = _strict_json_object(
        _read_bounded_regular_file(
            snapshot_path / "snapshot.json",
            _MAX_PRODUCT_JSON_BYTES,
            "current Agent Snapshot manifest",
        ),
        "current Agent Snapshot manifest",
    )
    session_payload = _strict_json_object(
        _read_bounded_regular_file(
            session_path / "state.json",
            _MAX_PRODUCT_JSON_BYTES,
            "current Agent Session binding",
        ),
        "current Agent Session binding",
    )
    if (
        pr_payload.get("provider") != "cli"
        or pr_payload.get("pr_number_or_external_review_id") != eval_input.task_id
        or snapshot_payload.get("pr_id") != pr_payload.get("pr_id")
        or session_payload.get("pr_id") != pr_payload.get("pr_id")
        or session_payload.get("snapshot_id") != snapshot_payload.get("snapshot_id")
    ):
        raise _CurrentArtifactError(
            "current Agent PRWorkspace locator binding is invalid"
        )
    try:
        store = PRWorkspaceStore(runtime_root)
        session = store.open_session(
            pr_id=pr_payload["pr_id"],
            snapshot_id=snapshot_payload["snapshot_id"],
            session_id=session_payload["session_id"],
        )
        repository_identity = canonical_repository_identity(
            RevisionResolver().repository_identity(workspace)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _CurrentArtifactError(
            "current Agent PRWorkspace cannot be opened"
        ) from error
    repository = repository_from_eval_input(eval_input)
    if (
        session.workspace.resolved_pr.repository != repository_identity
        or session.snapshot.base_sha != repository.base_revision
        or session.snapshot.head_sha != repository.head_revision
    ):
        raise _CurrentArtifactError(
            "current Agent Snapshot authority does not match the Trial"
        )
    try:
        manifest = SessionV6Store(store, session).load()
    except (TypeError, ValueError) as error:
        raise _CurrentArtifactError(
            "current Agent Session v6 manifest is invalid"
        ) from error
    if (
        manifest.status is not RunStatus.COMPLETED
        or any(
            checkpoint.status is not PhaseStatus.COMPLETED
            for checkpoint in manifest.phases.values()
        )
    ):
        raise _CurrentArtifactError(
            "current Agent Session v6 is not terminal-completed"
        )
    try:
        bundle = store.load_review_result_bundle(session.snapshot)
        if bundle is None:
            raise ValueError("missing ReviewResult")
        review_result = ProductReviewResultV2.from_json(
            bundle.review_result_bytes
        )
    except (TypeError, ValueError) as error:
        raise _CurrentArtifactError(
            "current Agent ReviewResult is invalid"
        ) from error
    if (
        review_result.pr_id != session.workspace.pr_id
        or review_result.snapshot_id != session.snapshot.snapshot_id
        or stdout != bundle.review_result_bytes + b"\n"
    ):
        raise _CurrentArtifactError(
            "current Agent public output or ReviewResult binding changed"
        )
    try:
        diff = DiffArtifactStore(store).load(session.snapshot)
        intent_ref = next(
            item
            for item in manifest.phases["intent"].artifacts
            if item.logical_name == "intent.packet"
        )
        intent = IntentVersionEnvelope.from_json(
            store.read_verified_artifact(
                session.snapshot,
                intent_ref.artifact_id,
            )
        )
    except (StopIteration, TypeError, ValueError) as error:
        raise _CurrentArtifactError(
            "current Agent Intent or DiffArtifact is invalid"
        ) from error
    if intent.source_snapshot_id != session.snapshot.snapshot_id:
        raise _CurrentArtifactError(
            "current Agent Intent Snapshot binding changed"
        )
    return _CurrentV6Run(
        runtime_root=runtime_root,
        workspace_store=store,
        session=session,
        review_result=review_result,
        diff=diff,
        intent=intent,
    )


def _safe_control_directory(path: Path, context: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise _CurrentAdapterError("%s could not be inspected" % context) from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT
    ):
        raise _CurrentAdapterError("%s is not a safe directory" % context)
    return metadata


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _read_created_ci_bundle(path: Path) -> bytes:
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or getattr(before, "st_file_attributes", 0) & _REPARSE_POINT
            or before.st_nlink != 1
            or before.st_size > MAX_EVAL_INPUT_BYTES
        ):
            raise _CurrentAdapterError(
                "current Agent CI evidence bundle is not a bounded regular file"
            )
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or not _same_identity(before, opened)
            ):
                raise _CurrentAdapterError(
                    "current Agent CI evidence bundle changed before verification"
                )
            chunks: List[bytes] = []
            remaining = MAX_EVAL_INPUT_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = os.lstat(path)
        if (
            len(data) > MAX_EVAL_INPUT_BYTES
            or len(data) != after.st_size
            or not _same_identity(opened, after)
            or not _same_identity(after, path_after)
            or path_after.st_nlink != 1
        ):
            raise _CurrentAdapterError(
                "current Agent CI evidence bundle changed during verification"
            )
        return data
    except _CurrentAdapterError:
        raise
    except OSError as error:
        raise _CurrentAdapterError(
            "current Agent CI evidence bundle could not be verified"
        ) from error


def _write_ci_evidence_bundle(workspace: Path, eval_input: EvalInput) -> Optional[str]:
    target = eval_input.review_target
    if not isinstance(target, RepositoryReviewTarget):
        raise _CurrentAdapterError("current Agent requires a Repository Target")
    evidence = target.review_request.existing_ci_evidence
    if not evidence:
        return None
    payload = {
        "schema_version": _CI_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "entries": [item.to_dict() for item in evidence],
    }
    data = canonical_json_bytes(payload)
    if len(data) > MAX_EVAL_INPUT_BYTES:
        raise _CurrentAdapterError(
            "current Agent CI evidence bundle exceeds the EvalInput byte limit"
        )

    review_root = workspace / ".review-agent"
    control_root = review_root / "eval-input"
    try:
        if os.path.lexists(review_root):
            review_metadata = _safe_control_directory(
                review_root, "current Agent control root"
            )
        else:
            os.mkdir(review_root, 0o700)
            review_metadata = _safe_control_directory(
                review_root, "current Agent control root"
            )
        if os.path.lexists(control_root):
            raise _CurrentAdapterError(
                "current Agent eval-input control root already exists"
            )
        os.mkdir(control_root, 0o700)
        control_metadata = _safe_control_directory(
            control_root, "current Agent eval-input control root"
        )

        bundle_digest = hashlib.sha256(data).hexdigest()
        bundle_name = (
            _CI_EVIDENCE_BUNDLE_PREFIX
            + bundle_digest
            + _CI_EVIDENCE_BUNDLE_SUFFIX
        )
        temporary = control_root / ("." + bundle_name + ".tmp")
        destination = control_root / bundle_name
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
        )
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        try:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        temporary_data = _read_created_ci_bundle(temporary)
        if (
            temporary_data != data
            or hashlib.sha256(temporary_data).digest()
            != hashlib.sha256(data).digest()
        ):
            raise _CurrentAdapterError(
                "current Agent CI evidence bundle failed byte verification"
            )
        os.link(temporary, destination, follow_symlinks=False)
        os.unlink(temporary)

        final_data = _read_created_ci_bundle(destination)
        if (
            final_data != data
            or canonical_json_bytes(json.loads(final_data.decode("utf-8"))) != data
            or hashlib.sha256(final_data).digest() != hashlib.sha256(data).digest()
        ):
            raise _CurrentAdapterError(
                "current Agent CI evidence bundle failed canonical verification"
            )
        if (
            not _same_identity(
                review_metadata,
                _safe_control_directory(review_root, "current Agent control root"),
            )
            or not _same_identity(
                control_metadata,
                _safe_control_directory(
                    control_root, "current Agent eval-input control root"
                ),
            )
        ):
            raise _CurrentAdapterError(
                "current Agent CI evidence control path changed during publication"
            )
    except _CurrentAdapterError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise _CurrentAdapterError(
            "current Agent CI evidence bundle could not be published safely"
        ) from error
    return (_CI_EVIDENCE_CONTROL_DIRECTORY / bundle_name).as_posix()


def _initial_argv(
    adapter: _CurrentAdapterConfiguration,
    eval_input: EvalInput,
    workspace: Path,
    runtime_root: Path,
    ci_evidence_file: Optional[str] = None,
) -> List[str]:
    target = eval_input.review_target
    if not isinstance(target, RepositoryReviewTarget):
        raise _CurrentAdapterError("current Agent requires a Repository Target")
    request = target.review_request
    repository = repository_from_eval_input(eval_input)
    argv = [
        *adapter.command,
        "review",
        *adapter.review_arguments,
        "--repo=" + str(workspace),
        "--base=" + repository.base_revision,
        "--head=" + repository.head_revision,
        "--external-review-id=" + eval_input.task_id,
        "--workspace-root=" + str(runtime_root),
        "--format=json",
    ]
    for option, value in (
        ("--title=", request.title),
        ("--description=", request.description),
        ("--intent=", request.user_intent),
        ("--focus=", request.review_focus),
    ):
        if value is not None:
            argv.append(option + value)
    argv.extend("--requirement=" + item for item in request.linked_requirements)
    argv.extend("--project-rule=" + item for item in request.project_rules)
    if ci_evidence_file is not None:
        argv.append("--ci-evidence-file=" + ci_evidence_file)
    return argv


class CurrentAgentAdapter:
    """Invoke and convert the current product through its public CLI/artifacts."""

    ADAPTER_KIND = CURRENT_AGENT_ADAPTER_KIND
    ADAPTER_VERSION = CURRENT_AGENT_ADAPTER_VERSION

    def __init__(
        self,
        *,
        process_runner: Callable[..., BoundedProcessResult] = run_bounded_process,
    ) -> None:
        self._process_runner = process_runner

    @staticmethod
    def compatibility(
        eval_input: EvalInput,
        config: AgentRunConfig,
    ) -> AdapterCompatibility:
        if not isinstance(eval_input, EvalInput) or not isinstance(
            config, AgentRunConfig
        ):
            raise TypeError("adapter compatibility requires canonical input/config")
        if (
            eval_input.review_target.kind is not ReviewTargetKind.REPOSITORY
            or eval_input.review_target.kind is not config.target_kind
        ):
            raise AgentAdapterIncompatibleError(
                AdapterIncompatibilityReason.TARGET_KIND
            )
        if config.adapter_capabilities != current_agent_capabilities():
            raise AgentAdapterIncompatibleError(
                AdapterIncompatibilityReason.CAPABILITY_MISMATCH
            )
        return AdapterCompatibility(unsupported=frozenset())

    def run(
        self,
        eval_input: EvalInput,
        workspace: Path,
        config: AgentRunConfig,
        clarification_channel: ClarificationChannel,
        *,
        target_access: TargetAccess,
        target_materialization_id: str,
        cancel_event: Optional[threading.Event] = None,
    ) -> EvalSubmission:
        return self._run_v6(
            eval_input,
            workspace,
            config,
            clarification_channel,
            target_access=target_access,
            target_materialization_id=target_materialization_id,
            cancel_event=cancel_event,
        )

    def _run_v6(
        self,
        eval_input: EvalInput,
        workspace: Path,
        config: AgentRunConfig,
        clarification_channel: ClarificationChannel,
        *,
        target_access: TargetAccess,
        target_materialization_id: str,
        cancel_event: Optional[threading.Event] = None,
    ) -> EvalSubmission:
        del clarification_channel
        if (
            not isinstance(eval_input, EvalInput)
            or not isinstance(config, AgentRunConfig)
            or eval_input.task_id != config.task_id
            or eval_input.digest() != config.eval_input_digest
        ):
            raise AgentAdapterError(
                FailureCode.SCHEMA_MISMATCH,
                "current Agent invocation does not match its Trial binding",
                retryable=False,
            )
        started = time.monotonic()
        try:
            if not isinstance(workspace, Path):
                raise _CurrentAdapterError("current Agent workspace is invalid")
            if not isinstance(target_access, TargetAccess):
                raise _CurrentAdapterError(
                    "current Agent TargetAccess is invalid"
                )
            if target_access.target_materialization_id != target_materialization_id:
                raise _CurrentAdapterError(
                    "current Agent TargetAccess identity drifted"
                )
            resolved_workspace = workspace.resolve(strict=True)
            if not resolved_workspace.is_dir():
                raise _CurrentAdapterError(
                    "current Agent workspace is not a directory"
                )
            adapter = _configuration(config)
            if not self.compatibility(eval_input, config).compatible:
                raise _CurrentAdapterError(
                    "current Agent input is incompatible with the product CLI"
                )
            runtime_root = resolved_workspace / ".ra-v6"
            if os.path.lexists(runtime_root):
                raise _CurrentAdapterError(
                    "current Agent requires a fresh private v6 Runtime root"
                )
            ci_evidence_file = _write_ci_evidence_bundle(
                resolved_workspace,
                eval_input,
            )
            environment = build_subprocess_environment(
                adapter.environment_allowlist
            )
            result = self._invoke(
                _initial_argv(
                    adapter,
                    eval_input,
                    resolved_workspace,
                    runtime_root,
                    ci_evidence_file,
                ),
                stdin_bytes=b"",
                workspace=resolved_workspace,
                environment=environment,
                deadline=started + float(config.timeout_seconds),
                remaining_output=config.max_output_bytes,
                cancel_event=cancel_event,
            )
            if result.failure_code is not None:
                return _failure(
                    eval_input=eval_input,
                    config=config,
                    target_materialization_id=target_materialization_id,
                    code=result.failure_code,
                    elapsed=time.monotonic() - started,
                    retryable=result.failure_code
                    in {FailureCode.TIMEOUT, FailureCode.PROCESS_KILLED},
                )
            if result.returncode not in (None, 0):
                code = (
                    FailureCode.PROCESS_KILLED
                    if returncode_was_killed(result.returncode)
                    else FailureCode.NON_ZERO_EXIT
                )
                return _failure(
                    eval_input=eval_input,
                    config=config,
                    target_materialization_id=target_materialization_id,
                    code=code,
                    elapsed=time.monotonic() - started,
                    retryable=code is FailureCode.PROCESS_KILLED,
                )
            if result.returncode is None:
                return _failure(
                    eval_input=eval_input,
                    config=config,
                    target_materialization_id=target_materialization_id,
                    code=FailureCode.UNKNOWN,
                    elapsed=time.monotonic() - started,
                    retryable=True,
                )
            run = _discover_v6_run(
                runtime_root=runtime_root,
                workspace=resolved_workspace,
                eval_input=eval_input,
                stdout=result.stdout,
            )
            intent = _submission_intent_v6(run.intent)
            findings, evidence = _findings_and_evidence_v6(
                run=run,
                eval_input=eval_input,
                target_materialization_id=target_materialization_id,
            )
            trace_ref = TraceRef(
                type=TraceType.LOCAL_PATH,
                value=runtime_root.relative_to(resolved_workspace).as_posix(),
            )
            submission = EvalSubmission(
                schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
                task_id=eval_input.task_id,
                agent_id=config.agent_id,
                trial_id=config.trial_id,
                eval_input_digest=config.eval_input_digest,
                target_materialization_id=target_materialization_id,
                status=SubmissionStatus.COMPLETED,
                intent=intent,
                review=SubmissionReview(
                    findings=findings,
                    uncertainties=run.review_result.uncertainties,
                ),
                evidence=evidence,
                usage=empty_usage(
                    elapsed_seconds=max(0.0, time.monotonic() - started)
                ),
                trace_ref=trace_ref,
                failure=None,
            )
            return validate_submission_trace(
                submission,
                workspace=resolved_workspace,
                max_trace_bytes=config.max_trace_bytes,
            )
        except AgentAdapterIncompatibleError:
            raise
        except AgentAdapterError as error:
            return _failure(
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
                code=(
                    error.code
                    if error.code
                    in {FailureCode.OUTPUT_OVERFLOW, FailureCode.SCHEMA_MISMATCH}
                    else FailureCode.ADAPTER_ERROR
                ),
                elapsed=time.monotonic() - started,
                retryable=error.retryable,
            )
        except _CurrentArtifactError:
            return _failure(
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
                code=FailureCode.SCHEMA_MISMATCH,
                elapsed=time.monotonic() - started,
                retryable=False,
            )
        except _CurrentAdapterError:
            return _failure(
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
                code=FailureCode.ADAPTER_ERROR,
                elapsed=time.monotonic() - started,
                retryable=False,
            )
        except Exception:
            return _failure(
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
                code=FailureCode.SCHEMA_MISMATCH,
                elapsed=time.monotonic() - started,
                retryable=False,
            )

    def _invoke(
        self,
        argv: Sequence[str],
        *,
        stdin_bytes: bytes,
        workspace: Path,
        environment: Mapping[str, str],
        deadline: float,
        remaining_output: int,
        cancel_event: Optional[threading.Event] = None,
    ) -> BoundedProcessResult:
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            return BoundedProcessResult(
                stdout=b"",
                returncode=None,
                failure_code=FailureCode.TIMEOUT,
                output_bytes=0,
            )
        if remaining_output <= 0:
            return BoundedProcessResult(
                stdout=b"",
                returncode=None,
                failure_code=FailureCode.OUTPUT_OVERFLOW,
                output_bytes=0,
            )
        kwargs = {
            "stdin_bytes": stdin_bytes,
            "workspace": workspace,
            "environment": environment,
            "timeout_seconds": remaining_time,
            "max_output_bytes": remaining_output,
        }
        if cancel_event is not None:
            kwargs["cancel_event"] = cancel_event
        return self._process_runner(argv, **kwargs)


def _submission_intent_v6(
    envelope: IntentVersionEnvelope,
) -> SubmissionIntent:
    packet = envelope.packet
    if packet.goal is None:
        status = IntentResult.INSUFFICIENT
        claims: tuple[SubmissionIntentClaim, ...] = ()
    else:
        status = (
            IntentResult.SUFFICIENT
            if packet.source is ProductIntentSourceV2.EXPLICIT
            else IntentResult.PARTIAL
        )
        source = (
            IntentClaimSource.EXPLICIT
            if packet.source is ProductIntentSourceV2.EXPLICIT
            else IntentClaimSource.INFERRED
        )
        claims = (
            SubmissionIntentClaim(
                claim_id=stable_id(
                    "intent-goal",
                    envelope.source_snapshot_id,
                    packet.goal,
                ),
                dimension=IntentDimension.GOAL,
                text=packet.goal,
                source=source,
            ),
        )
    return SubmissionIntent(
        status=status,
        goal=packet.goal,
        acceptance_criteria=(),
        scope=(),
        constraints=(),
        claims=claims,
        clarification_questions=(),
        uncertainties=packet.uncertainties,
    )


def _fit_diff_excerpt(raw: bytes, *, line: int, new_start: int) -> str:
    if len(raw) <= MAX_EVIDENCE_EXCERPT_BYTES:
        try:
            return raw.decode("utf-8", "strict")
        except UnicodeError as error:
            raise _CurrentArtifactError(
                "current Agent Diff hunk is not UTF-8"
            ) from error
    lines = raw.splitlines(keepends=True)
    target_index = 0
    current = new_start
    for index, value in enumerate(lines):
        if value.startswith((b"+", b" ")) and not value.startswith(b"+++"):
            if current == line:
                target_index = index
                break
            current += 1
    start = max(0, target_index - 100)
    end = min(len(lines), target_index + 101)
    selected = b"".join(lines[start:end])
    prefix = b"...[earlier diff lines omitted]...\n" if start else b""
    suffix = b"\n...[later diff lines omitted]..." if end < len(lines) else b""
    budget = MAX_EVIDENCE_EXCERPT_BYTES - len(prefix) - len(suffix)
    selected = selected[: max(0, budget)]
    while selected:
        try:
            text = selected.decode("utf-8", "strict")
            break
        except UnicodeDecodeError as error:
            selected = selected[: error.start]
    else:
        text = ""
    return prefix.decode("ascii") + text + suffix.decode("ascii")


def _findings_and_evidence_v6(
    *,
    run: _CurrentV6Run,
    eval_input: EvalInput,
    target_materialization_id: str,
) -> tuple[tuple[SubmissionFinding, ...], tuple[SubmissionEvidence, ...]]:
    repository = repository_from_eval_input(eval_input)
    diff_store = DiffArtifactStore(run.workspace_store)
    findings: list[SubmissionFinding] = []
    evidence: list[SubmissionEvidence] = []
    severity_mapping = {
        ProductFindingSeverityV2.LOW: FindingSeverity.LOW,
        ProductFindingSeverityV2.MEDIUM: FindingSeverity.MEDIUM,
        ProductFindingSeverityV2.HIGH: FindingSeverity.HIGH,
        ProductFindingSeverityV2.BLOCKER: FindingSeverity.CRITICAL,
    }
    for finding in run.review_result.findings:
        file_match = next(
            (
                (file_index, file_entry)
                for file_index, file_entry in enumerate(run.diff.index.files)
                if file_entry.path == finding.path
            ),
            None,
        )
        if file_match is None:
            raise _CurrentArtifactError(
                "current Agent Finding path is outside DiffArtifact"
            )
        file_index, file_entry = file_match
        hunk_match = next(
            (
                (hunk_index, hunk)
                for hunk_index, hunk in enumerate(file_entry.hunks)
                if hunk.new_count > 0
                and hunk.new_start
                <= finding.line
                < hunk.new_start + hunk.new_count
            ),
            None,
        )
        if hunk_match is None:
            raise _CurrentArtifactError(
                "current Agent Finding line is outside DiffArtifact"
            )
        hunk_index, hunk = hunk_match
        excerpt = _fit_diff_excerpt(
            diff_store.read_hunk(run.diff, file_index, hunk_index),
            line=finding.line,
            new_start=hunk.new_start,
        )
        excerpt_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        evidence_id = stable_id(
            "diff-evidence",
            finding.finding_id,
            finding.path,
            finding.line,
            excerpt_hash,
        )
        evidence.append(
            SubmissionEvidence(
                evidence_id=evidence_id,
                source=RepositoryDiffEvidenceSource(
                    kind=EvidenceKind.REPOSITORY_DIFF,
                    target_materialization_id=target_materialization_id,
                    base_revision=repository.base_revision,
                    head_revision=repository.head_revision,
                    path=finding.path,
                ),
                content_hash=excerpt_hash,
                excerpt=excerpt,
            )
        )
        findings.append(
            SubmissionFinding(
                finding_id=finding.finding_id,
                claim=finding.claim,
                severity=severity_mapping[finding.severity],
                path=finding.path,
                side=DiffSide.RIGHT,
                from_line=finding.line,
                to_line=finding.line,
                evidence_refs=(evidence_id,),
                suggested_action=finding.suggestion,
            )
        )
    return tuple(findings), tuple(evidence)


__all__ = [
    "CURRENT_AGENT_ADAPTER_KIND",
    "CURRENT_AGENT_ADAPTER_VERSION",
    "CurrentAgentAdapter",
    "current_agent_capabilities",
]
