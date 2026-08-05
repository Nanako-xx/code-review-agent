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

from review_agent.artifacts import artifact_schema
from review_agent.brief import ReviewBrief
from review_agent.execution_profile import AgentExecutionProfile
from review_agent.hydration import (
    clarification_questions_from_dict,
    intent_claims_from_dict,
    review_brief_from_dict,
)
from review_agent.models import (
    ClarificationQuestion as ProductClarificationQuestion,
    ClarificationStatus as ProductClarificationStatus,
    IntentClaim as ProductIntentClaim,
    IntentClaimState as ProductIntentClaimState,
    IntentField as ProductIntentField,
    IntentSource as ProductIntentSource,
)
from review_agent.observations import Observation, ObservationStore
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import SESSION_SCHEMA_VERSION, SessionManifest
from review_agent.session_store import SessionStore

from ..clarification import (
    UNANSWERED_CLARIFICATION_CONTINUE,
    ClarificationChannel,
    ClarificationProtocolError,
    unanswered_clarification_action,
)
from ..config import ClarificationMatcherSnapshot
from ..config import AdapterCapabilitiesV2
from ..artifacts import TargetAccess
from ..models import (
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    MAX_EVAL_INPUT_BYTES,
    ClarificationAction,
    EvalInput,
    EvalSubmission,
    EvidenceKind,
    FailureCode,
    FindingSeverity,
    IntentClaimSource,
    IntentDimension,
    IntentResult,
    ReviewTargetKind,
    RepositoryDiffEvidenceSource,
    RepositoryFileEvidenceSource,
    RepositoryReviewTarget,
    SubmissionClarificationExchange,
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
CURRENT_AGENT_ADAPTER_VERSION = "2"


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
        "memory_mode",
    }
)
_ENVIRONMENT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_RUN_ID_RE = re.compile(r"^review-[0-9a-f]{12}$")
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
        "--memory-mode",
        "--memory-root",
        "--non-interactive",
    }
)


class _CurrentAdapterError(RuntimeError):
    pass


class _CurrentArtifactError(_CurrentAdapterError):
    pass


@dataclass(frozen=True)
class _CurrentAdapterConfiguration:
    command: Tuple[str, ...]
    review_arguments: Tuple[str, ...]
    environment_allowlist: Tuple[str, ...]
    memory_mode: str
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
    loop_arguments = [
        argument
        for argument in review_arguments
        if argument.startswith("--reviewer-loop")
    ]
    if loop_arguments != ["--reviewer-loop=agent-loop"]:
        raise _CurrentAdapterError(
            "current Agent must use exactly one frozen agent-loop argument"
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
    memory_mode = raw["memory_mode"]
    if memory_mode not in {"off", "read", "read-write"}:
        raise _CurrentAdapterError("current Agent memory mode is invalid")
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
    return _CurrentAdapterConfiguration(
        command=command,
        review_arguments=review_arguments,
        environment_allowlist=tuple(environment),
        memory_mode=memory_mode,
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


def _safe_run_directories(workspace: Path) -> Dict[str, Path]:
    root = workspace / ".review-agent" / "runs"
    if not os.path.lexists(root):
        return {}
    info = os.lstat(root)
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    ):
        raise _CurrentAdapterError("current Agent run root is unsafe")
    result: Dict[str, Path] = {}
    with os.scandir(root) as entries:
        for entry in entries:
            if len(result) >= 128:
                raise _CurrentAdapterError("current Agent run root is over its limit")
            entry_info = os.lstat(entry.path)
            if (
                not stat.S_ISDIR(entry_info.st_mode)
                or stat.S_ISLNK(entry_info.st_mode)
                or getattr(entry_info, "st_file_attributes", 0) & _REPARSE_POINT
                or _RUN_ID_RE.fullmatch(entry.name) is None
            ):
                raise _CurrentAdapterError(
                    "current Agent run root contains an unsafe entry"
                )
            result[entry.name] = Path(entry.path)
    return result


def _discover_new_run(before: Mapping[str, Path], workspace: Path) -> Path:
    after = _safe_run_directories(workspace)
    names = sorted(set(after) - set(before))
    if len(names) != 1:
        raise _CurrentAdapterError("current Agent did not create exactly one run")
    return after[names[0]]


def _path_equal(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(
        str(Path(right).resolve())
    )


def _load_session(
    run_dir: Path,
    *,
    workspace: Path,
    eval_input: EvalInput,
    execution_profile: AgentExecutionProfile,
) -> Tuple[SessionStore, SessionManifest]:
    store = SessionStore(run_dir)
    manifest = store.load()
    repository = repository_from_eval_input(eval_input)
    try:
        Path(manifest.repository.git_common_dir).resolve().relative_to(
            workspace.resolve()
        )
    except (OSError, ValueError) as exc:
        raise _CurrentArtifactError(
            "current Agent Git authority escapes the Trial workspace"
        ) from exc
    if (
        manifest.schema_version != SESSION_SCHEMA_VERSION
        or manifest.review_id != run_dir.name
        or manifest.root_review_id != manifest.review_id
        or manifest.parent_review_id is not None
        or not _path_equal(manifest.repository.canonical_path, workspace)
        or manifest.revisions.requested_base != repository.base_revision
        or manifest.revisions.requested_head != repository.head_revision
        or manifest.revisions.resolved_base_sha != repository.base_revision
        or manifest.revisions.resolved_head_sha != repository.head_revision
    ):
        raise _CurrentArtifactError("current Agent Session binding is invalid")
    try:
        actual_profile = AgentExecutionProfile.from_execution(manifest.execution)
    except (TypeError, ValueError) as exc:
        raise _CurrentArtifactError(
            "current Agent Session execution profile is invalid"
        ) from exc
    if (
        actual_profile.digest() != execution_profile.digest()
        or actual_profile.to_dict() != execution_profile.to_dict()
    ):
        raise AgentAdapterIncompatibleError(
            AdapterIncompatibilityReason.EXECUTION_PROFILE_MISMATCH
        )
    return store, manifest


def _read_bounded_regular_file(path: Path, maximum: int, context: str) -> bytes:
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or getattr(before, "st_file_attributes", 0) & _REPARSE_POINT
            or before.st_nlink != 1
            or before.st_size > maximum
        ):
            raise _CurrentArtifactError("%s is not a bounded regular file" % context)
        with path.open("rb") as handle:
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


_FIELD_TO_DIMENSION = {
    ProductIntentField.GOAL: IntentDimension.GOAL,
    ProductIntentField.ACCEPTANCE_CRITERIA: IntentDimension.ACCEPTANCE_CRITERION,
    ProductIntentField.SCOPE: IntentDimension.SCOPE,
    ProductIntentField.CONSTRAINTS: IntentDimension.CONSTRAINT,
}


def _submission_claims(
    claims: Iterable[ProductIntentClaim],
) -> Tuple[SubmissionIntentClaim, ...]:
    result = []
    for claim in claims:
        if claim.claim_state is not ProductIntentClaimState.ACTIVE:
            continue
        source = (
            IntentClaimSource.EXPLICIT
            if claim.source is ProductIntentSource.EXPLICIT
            else IntentClaimSource.INFERRED
        )
        result.append(
            SubmissionIntentClaim(
                claim_id=claim.claim_id,
                dimension=_FIELD_TO_DIMENSION[claim.field],
                text=claim.value,
                source=source,
            )
        )
    return tuple(result)


def _intent_from_candidates(
    claims: Sequence[ProductIntentClaim],
    uncertainties: Sequence[str],
    transcript: Sequence[SubmissionClarificationExchange],
) -> SubmissionIntent:
    active = [item for item in claims if item.claim_state is ProductIntentClaimState.ACTIVE]
    values = {
        field: tuple(item.value for item in active if item.field is field)
        for field in ProductIntentField
    }
    goals = values[ProductIntentField.GOAL]
    extra_uncertainties = list(uncertainties)
    if len(goals) > 1:
        extra_uncertainties.append("Multiple active goal candidates remain unresolved")
    return SubmissionIntent(
        status=IntentResult.INSUFFICIENT,
        goal=goals[0] if len(goals) == 1 else None,
        acceptance_criteria=values[ProductIntentField.ACCEPTANCE_CRITERIA],
        scope=values[ProductIntentField.SCOPE],
        constraints=values[ProductIntentField.CONSTRAINTS],
        claims=_submission_claims(active),
        clarification_questions=tuple(transcript),
        uncertainties=tuple(dict.fromkeys(extra_uncertainties)),
    )


def _load_intent_discovery(
    run_dir: Path,
    store: SessionStore,
    manifest: SessionManifest,
    revision: str,
) -> Tuple[List[ProductIntentClaim], List[ProductClarificationQuestion], List[str]]:
    candidates_payload = _load_registered_json(
        run_dir=run_dir,
        store=store,
        manifest=manifest,
        name="intent_candidates",
        expected_path="intent_candidates.json",
        expected_phase=RunPhase.INTENT_DISCOVERY,
        expected_revision=revision,
    )
    questions_payload = _load_registered_json(
        run_dir=run_dir,
        store=store,
        manifest=manifest,
        name="intent_questions",
        expected_path="intent_questions.json",
        expected_phase=RunPhase.INTENT_DISCOVERY,
        expected_revision=revision,
    )
    claims = intent_claims_from_dict(candidates_payload)
    questions = clarification_questions_from_dict(questions_payload)
    uncertainties = candidates_payload.get("uncertainties", [])
    if not isinstance(uncertainties, list) or any(
        type(item) is not str or not item for item in uncertainties
    ):
        raise _CurrentArtifactError("intent candidate uncertainties are invalid")
    return claims, questions, list(uncertainties)


def _material_claim(
    question: ProductClarificationQuestion,
    claims: Sequence[ProductIntentClaim],
    matcher_snapshot: ClarificationMatcherSnapshot,
) -> str:
    by_id = {claim.claim_id: claim.value for claim in claims}
    claim_values = [by_id[item] for item in question.claim_ids if item in by_id]
    if question.claim_ids:
        if len(claim_values) == len(question.claim_ids):
            return " | ".join(claim_values)
        if matcher_snapshot.matcher_id == "canonical-material-claim":
            raise AgentAdapterIncompatibleError(
                AdapterIncompatibilityReason.CANONICAL_MATERIAL_CLAIM_UNAVAILABLE
            )
    if question.proposed_values:
        return " | ".join(question.proposed_values)
    if matcher_snapshot.matcher_id == "canonical-material-claim":
        raise AgentAdapterIncompatibleError(
            AdapterIncompatibilityReason.CANONICAL_MATERIAL_CLAIM_UNAVAILABLE
        )
    return question.question


def _ask_clarification(
    question: ProductClarificationQuestion,
    claims: Sequence[ProductIntentClaim],
    config: AgentRunConfig,
    clarification_channel: ClarificationChannel,
) -> SubmissionClarificationExchange:
    dimension = _FIELD_TO_DIMENSION[question.field]
    try:
        material_claim = _material_claim(
            question,
            claims,
            config.clarification_matcher,
        )
    except AgentAdapterIncompatibleError as exc:
        if (
            exc.reason
            is not AdapterIncompatibilityReason.CANONICAL_MATERIAL_CLAIM_UNAVAILABLE
            or unanswered_clarification_action(config.agent.parameters)
            != UNANSWERED_CLARIFICATION_CONTINUE
        ):
            raise
        return clarification_channel.skip_unresolved(
            question_id=question.question_id,
            dimension=dimension,
            question=question.question,
            proposed_values=tuple(question.proposed_values),
        )
    return clarification_channel.ask(
        question_id=question.question_id,
        dimension=dimension,
        question=question.question,
        material_claim=material_claim,
        proposed_values=tuple(question.proposed_values),
    )


def _answer_input(
    exchange: SubmissionClarificationExchange,
    question: ProductClarificationQuestion,
) -> Optional[bytes]:
    action = exchange.action
    if action is None or action is ClarificationAction.DEFER:
        return None
    if action is ClarificationAction.CONFIRM:
        if exchange.matched_answer_id is None:
            return b"confirm:benchmark-auto-accept\n"
        return b"confirm\n"
    if action is ClarificationAction.REJECT:
        if not question.claim_ids:
            raise _CurrentAdapterError("reject cannot be represented by product CLI")
        return b"reject\n"
    if action is ClarificationAction.SKIP:
        if exchange.matched_answer_id is None:
            return b"continue-with-uncertainty:benchmark-no-user\n"
        return b"skip\n"
    if action is ClarificationAction.CORRECT:
        if exchange.response is None:
            raise _CurrentAdapterError("correct answer has no product response")
        parsed = tuple(
            item.strip() for item in exchange.response.split(";") if item.strip()
        )
        if parsed != exchange.resolved_values:
            raise _CurrentAdapterError(
                "correct answer cannot be represented losslessly by product CLI"
            )
        return ("correct\n" + exchange.response + "\n").encode("utf-8")
    raise _CurrentAdapterError("unsupported clarification action")


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
    memory_root: Path,
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
        "--memory-mode=" + adapter.memory_mode,
        "--memory-root=" + str(memory_root),
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


def _resume_argv(
    adapter: _CurrentAdapterConfiguration,
    review_id: str,
    workspace: Path,
) -> List[str]:
    return [
        *adapter.command,
        "resume",
        review_id,
        "--repo=" + str(workspace),
    ]


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
        transcript: List[SubmissionClarificationExchange] = []
        run_dir: Optional[Path] = None
        try:
            if not isinstance(workspace, Path):
                raise _CurrentAdapterError("current Agent workspace is invalid")
            if not isinstance(target_access, TargetAccess):
                raise _CurrentAdapterError("current Agent TargetAccess is invalid")
            if target_access.target_materialization_id != target_materialization_id:
                raise _CurrentAdapterError("current Agent TargetAccess identity drifted")
            resolved_workspace = workspace.resolve(strict=True)
            if not resolved_workspace.is_dir():
                raise _CurrentAdapterError("current Agent workspace is not a directory")
            adapter = _configuration(config)
            if not self.compatibility(eval_input, config).compatible:
                raise _CurrentAdapterError(
                    "current Agent input is incompatible with the product CLI"
                )
            before = _safe_run_directories(resolved_workspace)
            if before:
                raise _CurrentAdapterError(
                    "current Agent requires a fresh isolated Trial workspace"
                )
            memory_root = (
                resolved_workspace / ".review-agent" / "eval-memory" / config.trial_id
            )
            ci_evidence_file = _write_ci_evidence_bundle(
                resolved_workspace,
                eval_input,
            )
            environment = build_subprocess_environment(
                adapter.environment_allowlist
            )
            remaining_output = config.max_output_bytes
            deadline = started + float(config.timeout_seconds)

            result = self._invoke(
                _initial_argv(
                    adapter,
                    eval_input,
                    resolved_workspace,
                    memory_root,
                    ci_evidence_file,
                ),
                stdin_bytes=b"",
                workspace=resolved_workspace,
                environment=environment,
                deadline=deadline,
                remaining_output=remaining_output,
                cancel_event=cancel_event,
            )
            remaining_output -= min(result.output_bytes, remaining_output)
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
            try:
                run_dir = _discover_new_run(before, resolved_workspace)
            except _CurrentAdapterError:
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
                raise
            store, manifest = _load_session(
                run_dir,
                workspace=resolved_workspace,
                eval_input=eval_input,
                execution_profile=adapter.execution_profile,
            )

            revision = (
                repository_from_eval_input(eval_input).base_revision
                + ".."
                + repository_from_eval_input(eval_input).head_revision
            )
            while manifest.status is RunStatus.AWAITING_USER:
                claims, questions, uncertainties = _load_intent_discovery(
                    run_dir, store, manifest, revision
                )
                asked_question_ids = {item.question_id for item in transcript}
                open_questions = [
                    item
                    for item in questions
                    if item.status
                    in {
                        ProductClarificationStatus.PENDING,
                        ProductClarificationStatus.OPEN,
                    }
                    and item.question_id not in asked_question_ids
                ]
                if not open_questions:
                    raise _CurrentAdapterError(
                        "awaiting Session has no new clarification question"
                    )
                question = open_questions[0]
                exchange = _ask_clarification(
                    question,
                    claims,
                    config,
                    clarification_channel,
                )
                transcript.append(exchange)
                answer = _answer_input(exchange, question)
                if answer is None:
                    intent = _intent_from_candidates(
                        claims, uncertainties, transcript
                    )
                    return _failure(
                        eval_input=eval_input,
                        config=config,
                        target_materialization_id=target_materialization_id,
                        code=FailureCode.CLARIFICATION_REQUIRED,
                        elapsed=time.monotonic() - started,
                        retryable=False,
                        intent=intent,
                        trace_ref=self._trace_ref(run_dir, resolved_workspace),
                        workspace=resolved_workspace,
                    )
                result = self._invoke(
                    _resume_argv(adapter, manifest.review_id, resolved_workspace),
                    stdin_bytes=answer,
                    workspace=resolved_workspace,
                    environment=environment,
                    deadline=deadline,
                    remaining_output=remaining_output,
                    cancel_event=cancel_event,
                )
                remaining_output -= min(result.output_bytes, remaining_output)
                if result.failure_code is not None:
                    return _failure(
                        eval_input=eval_input,
                        config=config,
                        target_materialization_id=target_materialization_id,
                        code=result.failure_code,
                        elapsed=time.monotonic() - started,
                        retryable=result.failure_code
                        in {FailureCode.TIMEOUT, FailureCode.PROCESS_KILLED},
                        intent=_intent_from_candidates(
                            claims, uncertainties, transcript
                        ),
                        trace_ref=self._trace_ref(run_dir, resolved_workspace),
                        workspace=resolved_workspace,
                    )
                store, manifest = _load_session(
                    run_dir,
                    workspace=resolved_workspace,
                    eval_input=eval_input,
                    execution_profile=adapter.execution_profile,
                )
                if (
                    result.returncode not in (None, 0)
                    and manifest.status is RunStatus.AWAITING_USER
                ):
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
                        intent=_intent_from_candidates(
                            claims, uncertainties, transcript
                        ),
                        trace_ref=self._trace_ref(
                            run_dir, resolved_workspace
                        ),
                        workspace=resolved_workspace,
                    )

            trace_ref = self._trace_ref(run_dir, resolved_workspace)
            if manifest.status is RunStatus.COMPLETED:
                submission = self._completed_submission(
                    eval_input=eval_input,
                    config=config,
                    target_materialization_id=target_materialization_id,
                    run_dir=run_dir,
                    store=store,
                    manifest=manifest,
                    transcript=tuple(transcript),
                    elapsed=time.monotonic() - started,
                    trace_ref=trace_ref,
                )
                return validate_submission_trace(
                    submission,
                    workspace=resolved_workspace,
                    max_trace_bytes=config.max_trace_bytes,
                )
            if manifest.status is RunStatus.FAILED:
                code = (
                    FailureCode.NON_ZERO_EXIT
                    if result.returncode not in (None, 0)
                    else FailureCode.UNKNOWN
                )
                return _failure(
                    eval_input=eval_input,
                    config=config,
                    target_materialization_id=target_materialization_id,
                    code=code,
                    elapsed=time.monotonic() - started,
                    retryable=False,
                    trace_ref=trace_ref,
                    workspace=resolved_workspace,
                )
            return _failure(
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
                code=FailureCode.UNKNOWN,
                elapsed=time.monotonic() - started,
                retryable=True,
                trace_ref=trace_ref,
                workspace=resolved_workspace,
            )
        except AgentAdapterIncompatibleError:
            raise
        except ClarificationProtocolError:
            return _failure(
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
                code=FailureCode.ADAPTER_ERROR,
                elapsed=time.monotonic() - started,
                retryable=False,
            )
        except AgentAdapterError as exc:
            return _failure(
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
                code=(
                    exc.code
                    if exc.code
                    in {
                        FailureCode.OUTPUT_OVERFLOW,
                        FailureCode.SCHEMA_MISMATCH,
                    }
                    else FailureCode.ADAPTER_ERROR
                ),
                elapsed=time.monotonic() - started,
                retryable=exc.retryable,
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
                code=FailureCode.SCHEMA_MISMATCH if run_dir else FailureCode.ADAPTER_ERROR,
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

    @staticmethod
    def _trace_ref(run_dir: Path, workspace: Path) -> TraceRef:
        return TraceRef(
            type=TraceType.LOCAL_PATH,
            value=run_dir.relative_to(workspace).as_posix(),
        )

    def _completed_submission(
        self,
        *,
        eval_input: EvalInput,
        config: AgentRunConfig,
        target_materialization_id: str,
        run_dir: Path,
        store: SessionStore,
        manifest: SessionManifest,
        transcript: Tuple[SubmissionClarificationExchange, ...],
        elapsed: float,
        trace_ref: TraceRef,
    ) -> EvalSubmission:
        repository = repository_from_eval_input(eval_input)
        revision = (
            repository.base_revision
            + ".."
            + repository.head_revision
        )
        brief_payload = _load_registered_json(
            run_dir=run_dir,
            store=store,
            manifest=manifest,
            name="review_brief",
            expected_path="review_brief.json",
            expected_phase=RunPhase.REPORTING,
            expected_revision=revision,
        )
        brief = review_brief_from_dict(brief_payload)
        if (
            brief.review_id != manifest.review_id
            or brief.base_revision != repository.base_revision
            or brief.head_revision != repository.head_revision
        ):
            raise _CurrentArtifactError("review brief identity is invalid")
        intent = _intent_from_brief(brief, transcript)
        findings = _findings_from_brief(brief)
        observations = _load_final_observations(
            run_dir=run_dir,
            store=store,
            manifest=manifest,
            revision=revision,
            config=config,
            eval_input=eval_input,
        )
        evidence = _evidence_from_observations(
            run_dir=run_dir,
            brief=brief,
            findings=findings,
            observations=observations,
            eval_input=eval_input,
            target_materialization_id=target_materialization_id,
        )
        return EvalSubmission(
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
                uncertainties=tuple(brief.uncertainties),
            ),
            evidence=evidence,
            usage=empty_usage(elapsed_seconds=elapsed),
            trace_ref=trace_ref,
            failure=None,
        )


def _intent_from_brief(
    brief: ReviewBrief,
    transcript: Tuple[SubmissionClarificationExchange, ...],
) -> SubmissionIntent:
    change = brief.change_intent
    assessment = brief.intent_assessment
    provenance = change.get("provenance", [])
    if not isinstance(provenance, list):
        raise _CurrentArtifactError("review brief intent provenance is invalid")
    claims = []
    for row in provenance:
        if not isinstance(row, Mapping) or row.get("claim_state") != "active":
            continue
        try:
            field = ProductIntentField(row["field"])
            source = IntentClaimSource(row["source"])
            claims.append(
                SubmissionIntentClaim(
                    claim_id=str(row["claim_id"]),
                    dimension=_FIELD_TO_DIMENSION[field],
                    text=str(row["value"]),
                    source=source,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _CurrentArtifactError("review brief intent claim is invalid") from exc
    history = assessment.get("clarification_history", [])
    if not isinstance(history, list) or len(history) != len(transcript):
        raise _CurrentArtifactError(
            "product clarification history does not match the channel"
        )
    if history:
        status_actions = {
            "pending": None,
            "open": None,
            "confirmed": ClarificationAction.CONFIRM,
            "corrected": ClarificationAction.CORRECT,
            "rejected": ClarificationAction.REJECT,
            "skipped": ClarificationAction.SKIP,
            "skipped_non_interactive": ClarificationAction.SKIP,
        }
        seen_history_ids = set()
        for exchange, row in zip(transcript, history):
            if not isinstance(row, Mapping):
                raise _CurrentArtifactError(
                    "product clarification history does not match the channel"
                )
            question_id = row.get("question_id")
            status = row.get("status")
            resolved_values = row.get("resolved_values")
            if (
                type(question_id) is not str
                or question_id in seen_history_ids
                or question_id != exchange.question_id
                or status not in status_actions
                or not isinstance(resolved_values, list)
            ):
                raise _CurrentArtifactError(
                    "product clarification history does not match the channel"
                )
            seen_history_ids.add(question_id)
            try:
                dimension = _FIELD_TO_DIMENSION[ProductIntentField(row["field"])]
            except (KeyError, TypeError, ValueError) as exc:
                raise _CurrentArtifactError(
                    "product clarification history does not match the channel"
                ) from exc
            if (
                row.get("question") != exchange.question
                or dimension is not exchange.dimension
                or status_actions[str(status)] is not exchange.action
                or tuple(resolved_values) != exchange.resolved_values
                or (
                    exchange.action is ClarificationAction.CORRECT
                    and row.get("user_response") != exchange.response
                )
            ):
                raise _CurrentArtifactError(
                    "product clarification result does not match the channel"
                )
    return SubmissionIntent(
        status=IntentResult(str(assessment["status"])),
        goal=change.get("goal"),
        acceptance_criteria=tuple(change.get("acceptance_criteria", [])),
        scope=tuple(change.get("scope", [])),
        constraints=tuple(change.get("constraints", [])),
        claims=tuple(claims),
        clarification_questions=transcript,
        uncertainties=tuple(assessment.get("uncertainties", [])),
    )


def _findings_from_brief(brief: ReviewBrief) -> Tuple[SubmissionFinding, ...]:
    result = []
    used = set()
    for index, finding in enumerate(brief.verified_findings):
        finding_id = finding.finding_id or stable_id(
            "finding",
            index,
            finding.claim,
            finding.severity,
            finding.path,
            finding.line,
        )
        if finding_id in used:
            finding_id = stable_id("finding", finding_id, index)
        used.add(finding_id)
        severity_mapping = {
            "low": FindingSeverity.LOW,
            "medium": FindingSeverity.MEDIUM,
            "high": FindingSeverity.HIGH,
            "critical": FindingSeverity.CRITICAL,
            "blocker": FindingSeverity.CRITICAL,
        }
        try:
            severity = severity_mapping[finding.severity.casefold()]
        except KeyError as exc:
            raise _CurrentArtifactError("review brief finding severity is invalid") from exc
        result.append(
            SubmissionFinding(
                finding_id=finding_id,
                claim=finding.claim,
                severity=severity,
                path=finding.path,
                side=None,
                from_line=finding.line,
                to_line=finding.line,
                evidence_refs=tuple(finding.evidence_refs),
                suggested_action=finding.suggested_action,
            )
        )
    return tuple(result)


def _load_final_observations(
    *,
    run_dir: Path,
    store: SessionStore,
    manifest: SessionManifest,
    revision: str,
    config: AgentRunConfig,
    eval_input: EvalInput,
) -> Dict[str, Observation]:
    descriptor = manifest.artifacts.get("observations")
    if (
        descriptor is None
        or descriptor.path != "observations.jsonl"
        or descriptor.schema != artifact_schema("observations")
        or descriptor.phase is not RunPhase.REPORTING
        or descriptor.revision_binding != revision
        or not store.validate_artifact(descriptor)
    ):
        raise _CurrentArtifactError("final observation descriptor is invalid")
    file_budget = min(
        config.budgets.max_execution_artifact_file_bytes,
        64 * 1024 * 1024,
    )
    total_budget = min(
        config.budgets.max_execution_artifact_total_bytes,
        512 * 1024 * 1024,
    )
    repository = repository_from_eval_input(eval_input)
    loaded = ObservationStore.load(
        run_dir,
        {
            revision,
            "base@" + repository.base_revision,
            "head@" + repository.head_revision,
        },
        max_log_bytes=file_budget,
        max_raw_artifact_bytes=file_budget,
        max_total_raw_bytes=total_budget,
    )
    return {item.observation_id: item for item in loaded.list_observations()}


def _read_observation_raw(run_dir: Path, observation: Observation) -> str:
    path = run_dir.joinpath(*PurePosixPath(observation.raw_artifact_ref).parts)
    try:
        raw = _read_bounded_regular_file(
            path,
            64 * 1024 * 1024,
            "observation raw artifact",
        )
        if hashlib.sha256(raw).hexdigest() != observation.content_hash:
            raise _CurrentArtifactError("observation raw artifact hash is invalid")
        text = raw.decode("utf-8", "strict")
    except _CurrentArtifactError:
        raise
    except (OSError, UnicodeError) as exc:
        raise _CurrentArtifactError("observation raw artifact is invalid") from exc
    return text


def _evidence_from_observations(
    *,
    run_dir: Path,
    brief: ReviewBrief,
    findings: Sequence[SubmissionFinding],
    observations: Mapping[str, Observation],
    eval_input: EvalInput,
    target_materialization_id: str,
) -> Tuple[SubmissionEvidence, ...]:
    referenced = {
        reference for finding in findings for reference in finding.evidence_refs
    }
    repository = repository_from_eval_input(eval_input)
    base = repository.base_revision
    head = repository.head_revision
    diff_revision = base + ".." + head
    result = []
    for evidence_id in sorted(referenced):
        observation = observations.get(evidence_id)
        if observation is None:
            continue
        if (
            observation.source == "git.read_range"
            and observation.revision in {"base@" + base, "head@" + head}
            and observation.path is not None
            and observation.line_start is not None
            and observation.line_end is not None
        ):
            raw = _read_observation_raw(run_dir, observation)
            result.append(
                SubmissionEvidence(
                    evidence_id=evidence_id,
                    source=RepositoryFileEvidenceSource(
                        kind=EvidenceKind.REPOSITORY_FILE,
                        target_materialization_id=target_materialization_id,
                        revision=observation.revision.split("@", 1)[1],
                        path=observation.path,
                        from_line=observation.line_start,
                        to_line=observation.line_end,
                    ),
                    content_hash=observation.content_hash,
                    excerpt=raw,
                )
            )
            continue
        if (
            observation.source == "git.compare_base_head"
            and observation.revision == diff_revision
            and observation.path is not None
        ):
            raw = _read_observation_raw(run_dir, observation)
            result.append(
                SubmissionEvidence(
                    evidence_id=evidence_id,
                    source=RepositoryDiffEvidenceSource(
                        kind=EvidenceKind.REPOSITORY_DIFF,
                        target_materialization_id=target_materialization_id,
                        base_revision=base,
                        head_revision=head,
                        path=observation.path,
                    ),
                    content_hash=observation.content_hash,
                    excerpt=raw,
                )
            )
            continue
        # Command output remains a dangling Finding reference until the Runner
        # can bind it to an immutable execution attestation.  Product artifacts
        # (especially truncated output) are not self-authenticating Evidence.
    return tuple(result)


__all__ = [
    "CURRENT_AGENT_ADAPTER_KIND",
    "CURRENT_AGENT_ADAPTER_VERSION",
    "CurrentAgentAdapter",
    "current_agent_capabilities",
]
