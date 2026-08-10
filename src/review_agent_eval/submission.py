"""Canonical Submission construction and adapter-boundary validation."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any, Iterable, Optional, Tuple

from .adapters.base import AgentAdapterError, AgentRunConfig
from .models import (
    EVAL_SUBMISSION_SCHEMA_VERSION,
    MAX_EVAL_SUBMISSION_BYTES,
    EvalInput,
    EvalSubmission,
    FailureCode,
    SchemaError,
    SubmissionClarificationExchange,
    SubmissionEvidence,
    SubmissionFailure,
    SubmissionIntent,
    SubmissionReview,
    SubmissionStatus,
    SubmissionUsage,
    TraceRef,
    TraceType,
    _strict_json_loads,
    submission_status_for_failure,
)


def _storage_path(path: Path) -> Path:
    """Use the extended-length namespace for Windows filesystem syscalls."""

    raw = os.fspath(path)
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return Path(raw)
    absolute = os.path.abspath(raw)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def empty_usage(*, elapsed_seconds: Any = None) -> SubmissionUsage:
    """Return an honest Usage object without estimating unavailable values."""

    return SubmissionUsage(
        elapsed_seconds=elapsed_seconds,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        tool_calls=None,
        cost_amount=None,
        cost_currency=None,
    )


def failure_submission(
    *,
    eval_input: EvalInput,
    config: AgentRunConfig,
    target_materialization_id: str,
    code: FailureCode,
    message: str,
    retryable: bool,
    intent: Optional[SubmissionIntent] = None,
    review: Optional[SubmissionReview] = None,
    evidence: Iterable[SubmissionEvidence] = (),
    usage: Optional[SubmissionUsage] = None,
    trace_ref: Optional[TraceRef] = None,
) -> EvalSubmission:
    """Build the one legal terminal status associated with ``code``."""

    if not isinstance(eval_input, EvalInput):
        raise TypeError("eval_input must be EvalInput")
    if not isinstance(config, AgentRunConfig):
        raise TypeError("config must be AgentRunConfig")
    if eval_input.task_id != config.task_id:
        raise SchemaError("failure submission eval_input task does not match config")
    if eval_input.digest() != config.eval_input_digest:
        raise SchemaError(
            "failure submission eval_input_digest does not match config"
        )
    status = submission_status_for_failure(code)
    if (
        status is SubmissionStatus.INVALID_OUTPUT
        or code is FailureCode.HARNESS_MATERIALIZATION_ERROR
    ):
        intent = None
        review = None
        evidence = ()
    return EvalSubmission(
        schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
        task_id=config.task_id,
        agent_id=config.agent_id,
        trial_id=config.trial_id,
        eval_input_digest=config.eval_input_digest,
        target_materialization_id=target_materialization_id,
        status=status,
        intent=intent,
        review=review,
        evidence=tuple(evidence),
        usage=empty_usage() if usage is None else usage,
        trace_ref=trace_ref,
        failure=SubmissionFailure(
            code=code,
            message=message,
            retryable=retryable,
        ),
    )


def validate_submission_binding(
    submission: EvalSubmission,
    *,
    eval_input: EvalInput,
    config: AgentRunConfig,
    target_materialization_id: str,
    clarification_transcript: Optional[
        Iterable[SubmissionClarificationExchange]
    ] = None,
) -> EvalSubmission:
    """Reject identity substitution and fabricated clarification history."""

    if not isinstance(submission, EvalSubmission):
        raise TypeError("submission must be EvalSubmission")
    if (
        submission.failure is not None
        and submission.failure.code
        is FailureCode.HARNESS_MATERIALIZATION_ERROR
    ):
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent output may not claim a Harness-owned failure code",
            retryable=False,
        )
    if (
        submission.task_id != eval_input.task_id
        or submission.task_id != config.task_id
    ):
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent output task identity does not match the invocation",
            retryable=False,
        )
    if (
        eval_input.digest() != config.eval_input_digest
        or submission.eval_input_digest != config.eval_input_digest
    ):
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent output input digest does not match the invocation",
            retryable=False,
        )
    if submission.target_materialization_id != target_materialization_id:
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent output target materialization does not match the invocation",
            retryable=False,
        )
    if submission.agent_id != config.agent_id:
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent output agent identity does not match the invocation",
            retryable=False,
        )
    if submission.trial_id != config.trial_id:
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent output trial identity does not match the invocation",
            retryable=False,
        )
    if any(
        item.source.kind not in config.adapter_capabilities.evidence_kinds
        for item in submission.evidence
    ):
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent output evidence kind was not declared by its Adapter",
            retryable=False,
        )

    trace_protocol = config.adapter_capabilities.trace_protocol
    if trace_protocol == "local-trace-v2":
        if (
            submission.trace_ref is not None
            and submission.trace_ref.type is not TraceType.LOCAL_PATH
        ):
            raise AgentAdapterError(
                FailureCode.SCHEMA_MISMATCH,
                "Agent output trace does not match its declared protocol",
                retryable=False,
            )
    elif trace_protocol == "none-v2":
        if submission.trace_ref is not None:
            raise AgentAdapterError(
                FailureCode.SCHEMA_MISMATCH,
                "Agent output trace is forbidden by its declared protocol",
                retryable=False,
            )
    else:
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent output trace protocol is unsupported",
            retryable=False,
        )

    expected = (
        ()
        if clarification_transcript is None
        else tuple(clarification_transcript)
    )
    actual: Tuple[SubmissionClarificationExchange, ...] = (
        ()
        if submission.intent is None
        else submission.intent.clarification_questions
    )
    clarification_protocol = config.adapter_capabilities.clarification_protocol
    if clarification_protocol == "none-v2":
        if actual or expected:
            raise AgentAdapterError(
                FailureCode.SCHEMA_MISMATCH,
                "Agent output clarification is forbidden by its declared protocol",
                retryable=False,
            )
    elif clarification_protocol == "canonical-clarification-v2":
        if actual != expected:
            raise AgentAdapterError(
                FailureCode.SCHEMA_MISMATCH,
                "Agent output clarification transcript does not match the channel",
                retryable=False,
            )
    else:
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent output clarification protocol is unsupported",
            retryable=False,
        )
    return submission


def parse_submission_output(
    data: bytes,
    *,
    eval_input: EvalInput,
    config: AgentRunConfig,
    target_materialization_id: str,
    clarification_transcript: Optional[
        Iterable[SubmissionClarificationExchange]
    ] = None,
) -> EvalSubmission:
    """Strictly parse one UTF-8 JSON Submission and enforce invocation binding."""

    if type(data) is not bytes:
        raise TypeError("Agent output must be bytes")
    maximum = min(config.max_output_bytes, MAX_EVAL_SUBMISSION_BYTES)
    if len(data) > maximum:
        raise AgentAdapterError(
            FailureCode.OUTPUT_OVERFLOW,
            "Agent output exceeds its byte limit",
            retryable=False,
        )
    try:
        payload = _strict_json_loads(
            data,
            MAX_EVAL_SUBMISSION_BYTES,
            "EvalSubmission JSON",
        )
    except SchemaError as exc:
        raise AgentAdapterError(
            FailureCode.INVALID_JSON,
            "Agent output is not one strict UTF-8 JSON document",
            retryable=False,
        ) from exc
    try:
        submission = EvalSubmission.from_dict(payload)
    except (SchemaError, UnicodeError, ValueError, RecursionError) as exc:
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent output does not satisfy EvalSubmission v2",
            retryable=False,
        ) from exc
    return validate_submission_binding(
        submission,
        eval_input=eval_input,
        config=config,
        target_materialization_id=target_materialization_id,
        clarification_transcript=clarification_transcript,
    )


def trace_path_within_workspace(trace_ref: TraceRef, workspace: Path) -> Path:
    """Resolve a relative local trace only beneath the Trial workspace."""

    from .models import TraceType

    if trace_ref.type is not TraceType.LOCAL_PATH:
        raise ValueError("trace_ref is not a local path")
    root = workspace.resolve(strict=True)
    candidate = Path(trace_ref.value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or candidate == Path(".")
        or ".." in candidate.parts
    ):
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent trace path must be relative to the Trial workspace",
            retryable=False,
        )
    current = root
    try:
        for part in candidate.parts:
            if part in {"", "."}:
                continue
            current = current / part
            info = os.lstat(_storage_path(current))
            attributes = getattr(info, "st_file_attributes", 0)
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if stat.S_ISLNK(info.st_mode) or attributes & reparse:
                raise AgentAdapterError(
                    FailureCode.SCHEMA_MISMATCH,
                    "Agent trace path contains a link or reparse point",
                    retryable=False,
                )
        resolved = current.resolve(strict=True)
    except AgentAdapterError:
        raise
    except OSError as exc:
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent trace path does not identify an existing safe object",
            retryable=False,
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AgentAdapterError(
            FailureCode.SCHEMA_MISMATCH,
            "Agent trace path escapes the Trial workspace",
            retryable=False,
        ) from exc
    return resolved


def validate_submission_trace(
    submission: EvalSubmission,
    *,
    workspace: Path,
    max_trace_bytes: int,
) -> EvalSubmission:
    """Validate a local trace tree without following attacker-created links."""

    from .models import TraceType

    trace_ref = submission.trace_ref
    if trace_ref is None or trace_ref.type is not TraceType.LOCAL_PATH:
        return submission
    if type(max_trace_bytes) is not int or max_trace_bytes < 1:
        raise TypeError("max_trace_bytes must be a positive integer")
    root = workspace.resolve(strict=True)
    trace_path = trace_path_within_workspace(trace_ref, root)
    stack = [trace_path]
    nodes = 0
    total_bytes = 0
    while stack:
        current = stack.pop()
        try:
            storage_current = _storage_path(current)
            info = os.lstat(storage_current)
        except OSError as exc:
            raise AgentAdapterError(
                FailureCode.SCHEMA_MISMATCH,
                "Agent trace path could not be inspected safely",
                retryable=False,
            ) from exc
        attributes = getattr(info, "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(info.st_mode) or attributes & reparse:
            raise AgentAdapterError(
                FailureCode.SCHEMA_MISMATCH,
                "Agent trace contains a link or reparse point",
                retryable=False,
            )
        nodes += 1
        if nodes > 100_000:
            raise AgentAdapterError(
                FailureCode.OUTPUT_OVERFLOW,
                "Agent trace exceeds its node limit",
                retryable=False,
            )
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(storage_current) as entries:
                    for entry in entries:
                        if nodes + len(stack) >= 100_000:
                            raise AgentAdapterError(
                                FailureCode.OUTPUT_OVERFLOW,
                                "Agent trace exceeds its node limit",
                                retryable=False,
                            )
                        stack.append(Path(entry.path))
            except AgentAdapterError:
                raise
            except OSError as exc:
                raise AgentAdapterError(
                    FailureCode.SCHEMA_MISMATCH,
                    "Agent trace directory could not be enumerated safely",
                    retryable=False,
                ) from exc
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise AgentAdapterError(
                FailureCode.SCHEMA_MISMATCH,
                "Agent trace contains an unsupported filesystem object",
                retryable=False,
            )
        total_bytes += info.st_size
        if total_bytes > max_trace_bytes:
            raise AgentAdapterError(
                FailureCode.OUTPUT_OVERFLOW,
                "Agent trace exceeds its byte limit",
                retryable=False,
            )
    return submission


__all__ = [
    "empty_usage",
    "failure_submission",
    "parse_submission_output",
    "trace_path_within_workspace",
    "validate_submission_binding",
    "validate_submission_trace",
]
