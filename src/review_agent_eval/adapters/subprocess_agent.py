"""Bounded one-shot adapter for canonical JSON subprocess agents."""

from __future__ import annotations

import errno
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

from ..clarification import ClarificationChannel
from ..models import EvalInput, EvalSubmission, FailureCode
from ..repository import _WindowsProcessJob, _resume_windows_process
from ..submission import (
    empty_usage,
    failure_submission,
    parse_submission_output,
    validate_submission_trace,
)
from .base import AdapterCompatibility, AgentAdapterError, AgentRunConfig


SUBPROCESS_JSON_ADAPTER_KIND = "subprocess-json-v1"

_ADAPTER_FIELDS = frozenset({"kind", "command", "environment_allowlist"})
_ENVIRONMENT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_PLACEHOLDER_RE = re.compile(r"\{(agent_id|task_id|trial_id|workspace)\}")
_FIXED_RUNTIME_ENVIRONMENT_KEYS = (
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
)
_MAX_COMMAND_ARGUMENTS = 256
_MAX_ARGUMENT_CHARS = 8_192
_MAX_ENVIRONMENT_KEYS = 128
_READ_CHUNK_BYTES = 16 * 1024
_POLL_SECONDS = 0.02
_THREAD_JOIN_SECONDS = 5.0
_PROCESS_WAIT_SECONDS = 5.0
_WINDOWS_CREATE_SUSPENDED = 0x00000004
_WINDOWS_KILLED_EXIT_CODES = frozenset(
    {
        0xC000013A,  # STATUS_CONTROL_C_EXIT
        0x40010004,  # DBG_TERMINATE_PROCESS
    }
)


class _AdapterConfigurationError(ValueError):
    """The frozen Agent snapshot does not contain the supported adapter shape."""


@dataclass(frozen=True)
class _SubprocessConfiguration:
    command: Tuple[str, ...]
    environment_allowlist: Tuple[str, ...]
    agent_snapshot_digest: str


@dataclass(frozen=True)
class BoundedProcessResult:
    stdout: bytes
    returncode: Optional[int]
    failure_code: Optional[FailureCode]
    output_bytes: int


class _OutputState:
    """Shared fixed-budget state for the two pipe reader threads."""

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.total = 0
        self.stdout_chunks: List[bytes] = []
        self.lock = threading.Lock()
        self.overflow = threading.Event()
        self.io_error = threading.Event()
        self.writer_error = threading.Event()
        self.stopping = threading.Event()
        self.wake = threading.Event()

    def consume(self, chunk: bytes, *, stdout: bool) -> None:
        with self.lock:
            remaining = max(0, self.maximum - self.total)
            if stdout and remaining:
                self.stdout_chunks.append(chunk[:remaining])
            if len(chunk) > remaining:
                self.total = self.maximum + 1
                self.overflow.set()
            else:
                self.total += len(chunk)
        self.wake.set()

    def stdout(self) -> bytes:
        with self.lock:
            return b"".join(self.stdout_chunks)


def _adapter_configuration(config: AgentRunConfig) -> _SubprocessConfiguration:
    """Parse only the adapter object frozen into this invocation's snapshot."""

    snapshot = config.agent
    snapshot_digest = snapshot.digest()
    raw = snapshot.parameters.get("adapter")
    if not isinstance(raw, Mapping):
        raise _AdapterConfigurationError("missing adapter object")
    if set(raw) != _ADAPTER_FIELDS:
        raise _AdapterConfigurationError("adapter object fields do not match v1")
    if raw["kind"] != SUBPROCESS_JSON_ADAPTER_KIND:
        raise _AdapterConfigurationError("unsupported adapter kind")

    raw_command = raw["command"]
    if isinstance(raw_command, (str, bytes)) or not isinstance(
        raw_command, Sequence
    ):
        raise _AdapterConfigurationError("adapter command must be an array")
    if not raw_command or len(raw_command) > _MAX_COMMAND_ARGUMENTS:
        raise _AdapterConfigurationError("adapter command size is invalid")
    command: List[str] = []
    for index, argument in enumerate(raw_command):
        if type(argument) is not str:
            raise _AdapterConfigurationError("adapter command contains a non-string")
        if len(argument) > _MAX_ARGUMENT_CHARS or "\x00" in argument:
            raise _AdapterConfigurationError("adapter command argument is invalid")
        remainder = _PLACEHOLDER_RE.sub("", argument)
        if "{" in remainder or "}" in remainder:
            raise _AdapterConfigurationError("adapter command placeholder is invalid")
        if index == 0:
            if remainder != argument:
                raise _AdapterConfigurationError("adapter executable is templated")
            if not argument or not Path(argument).is_absolute():
                raise _AdapterConfigurationError("adapter executable is not absolute")
        command.append(argument)

    raw_environment = raw["environment_allowlist"]
    if isinstance(raw_environment, (str, bytes)) or not isinstance(
        raw_environment, Sequence
    ):
        raise _AdapterConfigurationError(
            "adapter environment_allowlist must be an array"
        )
    if len(raw_environment) > _MAX_ENVIRONMENT_KEYS:
        raise _AdapterConfigurationError("adapter environment allowlist is too large")
    environment: List[str] = []
    seen = set()
    for key in raw_environment:
        if type(key) is not str or _ENVIRONMENT_KEY_RE.fullmatch(key) is None:
            raise _AdapterConfigurationError("adapter environment key is invalid")
        folded = key.casefold()
        if folded in seen:
            raise _AdapterConfigurationError("adapter environment key is duplicated")
        seen.add(folded)
        environment.append(key)

    # AgentConfigSnapshot is recursively frozen.  Rechecking the digest keeps
    # launch configuration coupled to the exact snapshot parsed above.
    if snapshot.digest() != snapshot_digest:
        raise _AdapterConfigurationError("agent snapshot changed during validation")
    return _SubprocessConfiguration(
        command=tuple(command),
        environment_allowlist=tuple(environment),
        agent_snapshot_digest=snapshot_digest,
    )


def _expand_command(
    adapter: _SubprocessConfiguration,
    *,
    config: AgentRunConfig,
    workspace: Path,
) -> List[str]:
    if config.agent.digest() != adapter.agent_snapshot_digest:
        raise _AdapterConfigurationError("agent snapshot does not match invocation")
    values = {
        "agent_id": config.agent_id,
        "task_id": config.task_id,
        "trial_id": config.trial_id,
        "workspace": str(workspace),
    }

    def substitute(match: re.Match[str]) -> str:
        return values[match.group(1)]

    return [_PLACEHOLDER_RE.sub(substitute, item) for item in adapter.command]


def build_subprocess_environment(allowlist: Sequence[str]) -> Dict[str, str]:
    """Copy only fixed Windows runtime keys and explicitly approved values."""

    result: Dict[str, str] = {}
    included = set()

    def inherit(requested: str) -> None:
        if os.name == "nt":
            actual = next(
                (
                    key
                    for key in os.environ
                    if key.casefold() == requested.casefold()
                ),
                None,
            )
        else:
            actual = requested if requested in os.environ else None
        if actual is None or actual.casefold() in included:
            return
        included.add(actual.casefold())
        result[actual] = os.environ[actual]

    for key in _FIXED_RUNTIME_ENVIRONMENT_KEYS:
        inherit(key)
    for key in allowlist:
        inherit(key)
    return result


def _read_pipe(stream: BinaryIO, state: _OutputState, *, stdout: bool) -> None:
    try:
        while not state.overflow.is_set():
            chunk = stream.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            if type(chunk) is not bytes:
                state.io_error.set()
                state.wake.set()
                break
            state.consume(chunk, stdout=stdout)
    except (OSError, ValueError):
        if not state.stopping.is_set():
            state.io_error.set()
            state.wake.set()
    except Exception:
        state.io_error.set()
        state.wake.set()
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _write_pipe(stream: BinaryIO, data: bytes, state: _OutputState) -> None:
    try:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = stream.write(view[offset : offset + _READ_CHUNK_BYTES])
            if written is None or written <= 0:
                raise OSError("stdin pipe made no progress")
            offset += written
    except BrokenPipeError:
        pass
    except OSError as exc:
        if not state.stopping.is_set() and not (
            exc.errno in {errno.EPIPE, errno.EINVAL}
            or getattr(exc, "winerror", None) in {6, 109, 232}
        ):
            state.writer_error.set()
            state.wake.set()
    except (ValueError, TypeError):
        if not state.stopping.is_set():
            state.writer_error.set()
            state.wake.set()
    except Exception:
        state.writer_error.set()
        state.wake.set()
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    job: _WindowsProcessJob,
) -> bool:
    """Terminate the isolated group/Job and reap its leader without waiting forever."""

    successful = True
    if os.name == "nt":
        try:
            job.terminate()
        except Exception:
            successful = False
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            successful = False

    try:
        process.wait(timeout=_PROCESS_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        successful = False
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=_PROCESS_WAIT_SECONDS)
        except (subprocess.TimeoutExpired, OSError):
            return False
    except OSError:
        successful = False
    return successful and process.poll() is not None


def _join_threads(threads: Sequence[threading.Thread]) -> bool:
    deadline = time.monotonic() + _THREAD_JOIN_SECONDS
    for thread in threads:
        if thread.ident is not None:
            thread.join(max(0.0, deadline - time.monotonic()))
    return not any(thread.is_alive() for thread in threads)


def _close_pipe(stream: Optional[BinaryIO]) -> None:
    if stream is None:
        return
    try:
        stream.close()
    except (OSError, ValueError):
        pass


def run_bounded_process(
    argv: Sequence[str],
    *,
    stdin_bytes: bytes,
    workspace: Path,
    environment: Mapping[str, str],
    timeout_seconds: Any,
    max_output_bytes: int,
) -> BoundedProcessResult:
    process: Optional[subprocess.Popen[bytes]] = None
    job: Optional[_WindowsProcessJob] = None
    state = _OutputState(max_output_bytes)
    threads: List[threading.Thread] = []
    returncode: Optional[int] = None
    terminal_failure: Optional[FailureCode] = None
    cleanup_ok = True

    try:
        job = _WindowsProcessJob()
        platform: Dict[str, Any] = {}
        if os.name == "nt":
            platform["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | _WINDOWS_CREATE_SUSPENDED
            )
            platform["start_new_session"] = False
        else:
            platform["start_new_session"] = True

        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(workspace),
            env=dict(environment),
            shell=False,
            close_fds=True,
            bufsize=0,
            **platform,
        )
        job.assign(process)
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise OSError("subprocess pipes were not created")

        threads = [
            threading.Thread(
                target=_read_pipe,
                args=(process.stdout, state),
                kwargs={"stdout": True},
                name="eval-subprocess-stdout",
                daemon=True,
            ),
            threading.Thread(
                target=_read_pipe,
                args=(process.stderr, state),
                kwargs={"stdout": False},
                name="eval-subprocess-stderr",
                daemon=True,
            ),
            threading.Thread(
                target=_write_pipe,
                args=(process.stdin, stdin_bytes, state),
                name="eval-subprocess-stdin",
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        _resume_windows_process(process)

        deadline = time.monotonic() + float(timeout_seconds)
        while True:
            if state.overflow.is_set():
                terminal_failure = FailureCode.OUTPUT_OVERFLOW
                break
            if state.io_error.is_set() or state.writer_error.is_set():
                terminal_failure = FailureCode.ADAPTER_ERROR
                break
            returncode = process.poll()
            if returncode is not None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminal_failure = FailureCode.TIMEOUT
                break
            state.wake.wait(min(_POLL_SECONDS, remaining))
            state.wake.clear()
    except Exception:
        terminal_failure = FailureCode.ADAPTER_ERROR
    finally:
        state.stopping.set()
        if process is not None and job is not None:
            if returncode is None:
                returncode = process.poll()
            cleanup_ok = _terminate_process_tree(process, job)
            if returncode is None:
                returncode = process.returncode

        threads_ok = _join_threads(threads)
        if not threads_ok and process is not None:
            _close_pipe(process.stdin)
            _close_pipe(process.stdout)
            _close_pipe(process.stderr)
            threads_ok = _join_threads(threads)
        cleanup_ok = cleanup_ok and threads_ok

        if process is not None:
            _close_pipe(process.stdin)
            _close_pipe(process.stdout)
            _close_pipe(process.stderr)
        if job is not None:
            try:
                job.close()
            except Exception:
                cleanup_ok = False

    if state.overflow.is_set():
        terminal_failure = FailureCode.OUTPUT_OVERFLOW
    elif (
        state.io_error.is_set()
        or state.writer_error.is_set()
        or not cleanup_ok
    ):
        terminal_failure = FailureCode.ADAPTER_ERROR
    return BoundedProcessResult(
        stdout=state.stdout(),
        returncode=returncode,
        failure_code=terminal_failure,
        output_bytes=state.total,
    )


def returncode_was_killed(returncode: int) -> bool:
    if returncode < 0:
        return True
    if os.name != "nt":
        return False
    unsigned = returncode & 0xFFFFFFFF
    return unsigned in _WINDOWS_KILLED_EXIT_CODES or unsigned == int(signal.SIGTERM)


_FAILURE_MESSAGES = {
    FailureCode.TIMEOUT: "Agent process exceeded its time limit",
    FailureCode.NON_ZERO_EXIT: "Agent process exited unsuccessfully",
    FailureCode.PROCESS_KILLED: "Agent process was killed",
    FailureCode.OUTPUT_OVERFLOW: "Agent process output exceeded its byte limit",
    FailureCode.INVALID_JSON: "Agent output was not valid UTF-8 JSON",
    FailureCode.SCHEMA_MISMATCH: "Agent output did not match the invocation schema",
    FailureCode.ADAPTER_ERROR: "Agent adapter boundary failed",
}


def _failed_submission(
    *,
    eval_input: EvalInput,
    config: AgentRunConfig,
    code: FailureCode,
    elapsed_seconds: float,
    retryable: bool,
) -> EvalSubmission:
    return failure_submission(
        eval_input=eval_input,
        config=config,
        code=code,
        message=_FAILURE_MESSAGES[code],
        retryable=retryable,
        usage=empty_usage(elapsed_seconds=max(0.0, elapsed_seconds)),
    )


class SubprocessAgentAdapter:
    """Run a snapshot-bound ``subprocess-json-v1`` Agent exactly once."""

    @staticmethod
    def compatibility(
        eval_input: EvalInput,
        config: AgentRunConfig,
    ) -> AdapterCompatibility:
        if not isinstance(eval_input, EvalInput) or not isinstance(
            config, AgentRunConfig
        ):
            raise TypeError("adapter compatibility requires canonical input/config")
        return AdapterCompatibility()

    def run(
        self,
        eval_input: EvalInput,
        workspace: Path,
        config: AgentRunConfig,
        clarification_channel: ClarificationChannel,
    ) -> EvalSubmission:
        del clarification_channel  # Generic v1 is one-shot and has no channel wire.
        started = time.monotonic()

        try:
            if not isinstance(eval_input, EvalInput):
                raise _AdapterConfigurationError("eval_input type is invalid")
            if not isinstance(config, AgentRunConfig):
                raise _AdapterConfigurationError("run config type is invalid")
            if eval_input.task_id != config.task_id:
                raise _AdapterConfigurationError("input task does not match run config")
            if eval_input.digest() != config.eval_input_digest:
                raise _AdapterConfigurationError(
                    "input content does not match run config"
                )
            if not isinstance(workspace, Path):
                raise _AdapterConfigurationError("workspace type is invalid")
            resolved_workspace = workspace.resolve(strict=True)
            if not resolved_workspace.is_dir():
                raise _AdapterConfigurationError("workspace is not a directory")
            adapter = _adapter_configuration(config)
            argv = _expand_command(
                adapter,
                config=config,
                workspace=resolved_workspace,
            )
            environment = build_subprocess_environment(
                adapter.environment_allowlist
            )
            stdin_bytes = eval_input.to_json().encode("utf-8")
        except _AdapterConfigurationError:
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                code=FailureCode.ADAPTER_ERROR,
                elapsed_seconds=time.monotonic() - started,
                retryable=False,
            )
        except Exception:
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                code=FailureCode.ADAPTER_ERROR,
                elapsed_seconds=time.monotonic() - started,
                retryable=True,
            )

        try:
            result = run_bounded_process(
                argv,
                stdin_bytes=stdin_bytes,
                workspace=resolved_workspace,
                environment=environment,
                timeout_seconds=config.timeout_seconds,
                max_output_bytes=config.max_output_bytes,
            )
        except Exception:
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                code=FailureCode.ADAPTER_ERROR,
                elapsed_seconds=time.monotonic() - started,
                retryable=True,
            )
        elapsed = time.monotonic() - started
        if result.failure_code is not None:
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                code=result.failure_code,
                elapsed_seconds=elapsed,
                retryable=result.failure_code
                in {
                    FailureCode.TIMEOUT,
                    FailureCode.PROCESS_KILLED,
                    FailureCode.ADAPTER_ERROR,
                },
            )

        if result.returncode is None:
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                code=FailureCode.ADAPTER_ERROR,
                elapsed_seconds=elapsed,
                retryable=True,
            )
        if result.returncode != 0:
            code = (
                FailureCode.PROCESS_KILLED
                if returncode_was_killed(result.returncode)
                else FailureCode.NON_ZERO_EXIT
            )
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                code=code,
                elapsed_seconds=elapsed,
                retryable=code is FailureCode.PROCESS_KILLED,
            )

        try:
            submission = parse_submission_output(
                result.stdout,
                eval_input=eval_input,
                config=config,
                clarification_transcript=(),
            )
            submission = validate_submission_trace(
                submission,
                workspace=resolved_workspace,
                max_trace_bytes=config.max_trace_bytes,
            )
        except AgentAdapterError as exc:
            code = (
                exc.code
                if exc.code
                in {
                    FailureCode.INVALID_JSON,
                    FailureCode.SCHEMA_MISMATCH,
                    FailureCode.OUTPUT_OVERFLOW,
                }
                else FailureCode.ADAPTER_ERROR
            )
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                code=code,
                elapsed_seconds=time.monotonic() - started,
                retryable=False,
            )
        except Exception:
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                code=FailureCode.ADAPTER_ERROR,
                elapsed_seconds=time.monotonic() - started,
                retryable=True,
            )

        return replace(
            submission,
            usage=empty_usage(elapsed_seconds=time.monotonic() - started),
        )


__all__ = [
    "BoundedProcessResult",
    "SUBPROCESS_JSON_ADAPTER_KIND",
    "SubprocessAgentAdapter",
    "build_subprocess_environment",
    "returncode_was_killed",
    "run_bounded_process",
]
