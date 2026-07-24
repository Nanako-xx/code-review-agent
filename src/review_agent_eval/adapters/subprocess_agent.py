"""Bounded one-shot adapter for canonical JSON subprocess agents."""

from __future__ import annotations

import errno
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Tuple

from ..artifacts import TargetAccess
from ..clarification import ClarificationChannel
from ..config import AdapterCapabilitiesV2
from ..models import (
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    EvalInput,
    EvalSubmission,
    EvidenceKind,
    FailureCode,
    ReviewTargetKind,
    canonical_json_bytes,
)
from ..repository import _WindowsProcessJob, _resume_windows_process
from ..submission import (
    empty_usage,
    failure_submission,
    parse_submission_output,
    validate_submission_trace,
)
from .base import (
    AdapterCompatibility,
    AdapterIncompatibilityReason,
    AgentAdapterError,
    AgentAdapterIncompatibleError,
    AgentRunConfig,
)


SUBPROCESS_JSON_ADAPTER_KIND = "subprocess-json-v2"
SUBPROCESS_JSON_ADAPTER_VERSION = "2"
SUBPROCESS_INVOCATION_SCHEMA_VERSION = "eval_subprocess_invocation_v2"
SUBPROCESS_WIRE_VERSION = "subprocess-json-v2"
SUBPROCESS_CLARIFICATION_PROTOCOL = "none-v2"
SUBPROCESS_TRACE_PROTOCOL = "local-trace-v2"
SUBPROCESS_ISOLATION_PROFILE = "target-workspace-v2"

_ADAPTER_FIELDS = frozenset(
    {"kind", "command", "environment_allowlist", "capabilities"}
)
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
_DESCENDANT_SCAN_SECONDS = 0.10
_DESCENDANT_FREEZE_ROUNDS = 4
_DESCENDANT_REAP_SECONDS = 1.0
_MAX_POSIX_PROCESS_ENTRIES = 100_000
_MAX_TRACKED_DESCENDANTS = 16_384
_MAX_PROC_STAT_BYTES = 4_096
_MAX_PS_OUTPUT_BYTES = 8 * 1024 * 1024
_PS_SNAPSHOT_TIMEOUT_SECONDS = 0.5
_WINDOWS_CREATE_SUSPENDED = 0x00000004
_WINDOWS_KILLED_EXIT_CODES = frozenset(
    {
        0xC000013A,  # STATUS_CONTROL_C_EXIT
        0x40010004,  # DBG_TERMINATE_PROCESS
    }
)


def subprocess_adapter_capabilities(
    *,
    target_kinds: Iterable[ReviewTargetKind] = tuple(ReviewTargetKind),
    evidence_kinds: Iterable[EvidenceKind] = tuple(EvidenceKind),
) -> AdapterCapabilitiesV2:
    """Build the explicit capability declaration for subprocess-json-v2."""

    return AdapterCapabilitiesV2.from_dict(
        {
            "schema_version": "eval_adapter_capabilities_v2",
            "adapter_id": SUBPROCESS_JSON_ADAPTER_KIND,
            "adapter_version": SUBPROCESS_JSON_ADAPTER_VERSION,
            "input_schema_version": EVAL_INPUT_SCHEMA_VERSION,
            "submission_schema_version": EVAL_SUBMISSION_SCHEMA_VERSION,
            "target_kinds": [item.value for item in target_kinds],
            "evidence_kinds": [item.value for item in evidence_kinds],
            "clarification_protocol": SUBPROCESS_CLARIFICATION_PROTOCOL,
            "trace_protocol": SUBPROCESS_TRACE_PROTOCOL,
            "subprocess_wire_version": SUBPROCESS_WIRE_VERSION,
            "isolation_profile": SUBPROCESS_ISOLATION_PROFILE,
        }
    )


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_json_value(item) for item in value]
    return value


class _AdapterConfigurationError(ValueError):
    """The frozen Agent snapshot does not contain the supported adapter shape."""


@dataclass(frozen=True)
class _SubprocessConfiguration:
    command: Tuple[str, ...]
    environment_allowlist: Tuple[str, ...]
    capabilities: AdapterCapabilitiesV2
    agent_snapshot_digest: str


@dataclass(frozen=True)
class BoundedProcessResult:
    stdout: bytes
    returncode: Optional[int]
    failure_code: Optional[FailureCode]
    output_bytes: int


@dataclass(frozen=True)
class _PosixProcessIdentity:
    pid: int
    parent_pid: int
    birth_marker: str
    state: str

    @property
    def terminated(self) -> bool:
        return self.state[:1].upper() in {"X", "Z"}


def _parse_linux_process_stat(data: bytes) -> Optional[_PosixProcessIdentity]:
    """Parse the identity fields whose positions follow Linux's ``comm`` field."""

    if not data or len(data) > _MAX_PROC_STAT_BYTES:
        return None
    opening = data.find(b"(")
    closing = data.rfind(b")")
    if opening <= 0 or closing <= opening or closing + 2 >= len(data):
        return None
    try:
        pid = int(data[:opening].strip())
        fields = data[closing + 2 :].split()
        if len(fields) < 20:
            return None
        state = fields[0].decode("ascii", errors="strict")
        parent_pid = int(fields[1])
        start_time = int(fields[19])
    except (UnicodeDecodeError, ValueError):
        return None
    if pid <= 0 or parent_pid < 0 or start_time < 0 or len(state) != 1:
        return None
    return _PosixProcessIdentity(
        pid=pid,
        parent_pid=parent_pid,
        birth_marker="proc:%d" % start_time,
        state=state,
    )


def _read_linux_process_snapshot(
    proc_root: Path = Path("/proc"),
) -> Optional[Tuple[Dict[int, _PosixProcessIdentity], bool]]:
    """Return a bounded Linux process snapshot, or ``None`` without procfs."""

    try:
        entries = os.scandir(proc_root)
    except OSError:
        return None
    processes: Dict[int, _PosixProcessIdentity] = {}
    complete = True
    with entries:
        for index, entry in enumerate(entries):
            if index >= _MAX_POSIX_PROCESS_ENTRIES:
                complete = False
                break
            if not entry.name.isdecimal():
                continue
            try:
                stat_path = Path(entry.path) / "stat"
                with stat_path.open("rb") as stream:
                    data = stream.read(_MAX_PROC_STAT_BYTES + 1)
            except OSError:
                # Processes can disappear between scandir and open. An inaccessible
                # unrelated process must not make every adapter cleanup fail.
                continue
            identity = _parse_linux_process_stat(data)
            if identity is None or identity.pid != int(entry.name):
                continue
            processes[identity.pid] = identity
    return processes, complete


def _read_ps_process_snapshot(
) -> Optional[Tuple[Dict[int, _PosixProcessIdentity], bool]]:
    """Best-effort process snapshot for POSIX hosts without Linux procfs."""

    executable = next(
        (
            candidate
            for candidate in ("/bin/ps", "/usr/bin/ps")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK)
        ),
        None,
    )
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "-axo", "pid=,ppid=,lstart=,state="],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_PS_SNAPSHOT_TIMEOUT_SECONDS,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            close_fds=True,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or len(completed.stdout) > _MAX_PS_OUTPUT_BYTES:
        return None

    processes: Dict[int, _PosixProcessIdentity] = {}
    complete = True
    for index, raw_line in enumerate(completed.stdout.splitlines()):
        if index >= _MAX_POSIX_PROCESS_ENTRIES:
            complete = False
            break
        fields = raw_line.decode("utf-8", errors="replace").split()
        if len(fields) < 4:
            continue
        try:
            pid = int(fields[0])
            parent_pid = int(fields[1])
        except ValueError:
            continue
        if pid <= 0 or parent_pid < 0:
            continue
        processes[pid] = _PosixProcessIdentity(
            pid=pid,
            parent_pid=parent_pid,
            birth_marker="ps:" + " ".join(fields[2:-1]),
            state=fields[-1],
        )
    return processes, complete


def _read_posix_process_snapshot(
) -> Optional[Tuple[Dict[int, _PosixProcessIdentity], bool]]:
    proc_snapshot = (
        _read_linux_process_snapshot()
        if sys.platform.startswith("linux")
        else None
    )
    if proc_snapshot is not None and proc_snapshot[1]:
        return proc_snapshot
    ps_snapshot = _read_ps_process_snapshot()
    if ps_snapshot is not None:
        return ps_snapshot
    return proc_snapshot


def _collect_posix_descendants(
    root_pid: int,
    processes: Mapping[int, _PosixProcessIdentity],
    *,
    additional_roots: Sequence[int] = (),
) -> Tuple[Dict[int, _PosixProcessIdentity], bool]:
    children: Dict[int, List[_PosixProcessIdentity]] = {}
    for identity in processes.values():
        children.setdefault(identity.parent_pid, []).append(identity)

    descendants: Dict[int, _PosixProcessIdentity] = {}
    pending = list(dict.fromkeys((root_pid, *additional_roots)))
    seen = set(pending)
    complete = True
    while pending:
        parent_pid = pending.pop()
        for identity in children.get(parent_pid, ()):
            if identity.pid in seen:
                continue
            if len(descendants) >= _MAX_TRACKED_DESCENDANTS:
                complete = False
                pending.clear()
                break
            seen.add(identity.pid)
            descendants[identity.pid] = identity
            pending.append(identity.pid)
    return descendants, complete


class _PosixDescendantTracker:
    """Remember descendants even after ``setsid`` and later re-parenting."""

    def __init__(self, root_pid: int) -> None:
        self.root_pid = root_pid
        self._known: Dict[int, str] = {}
        self.complete = True

    def observe(self) -> Tuple[_PosixProcessIdentity, ...]:
        snapshot = _read_posix_process_snapshot()
        if snapshot is None:
            if self._known:
                self.complete = False
            return ()
        processes, snapshot_complete = snapshot
        known_roots = tuple(
            pid
            for pid, marker in self._known.items()
            if (identity := processes.get(pid)) is not None
            and identity.birth_marker == marker
            and not identity.terminated
        )
        descendants, tree_complete = _collect_posix_descendants(
            self.root_pid,
            processes,
            additional_roots=known_roots,
        )
        self.complete = self.complete and snapshot_complete and tree_complete
        for identity in descendants.values():
            if (
                identity.pid not in self._known
                and len(self._known) >= _MAX_TRACKED_DESCENDANTS
            ):
                self.complete = False
                continue
            self._known[identity.pid] = identity.birth_marker

        return tuple(
            identity
            for pid, marker in self._known.items()
            if (identity := processes.get(pid)) is not None
            and identity.birth_marker == marker
            and not identity.terminated
        )


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
        raise _AdapterConfigurationError("adapter object fields do not match v2")
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

    try:
        capabilities = AdapterCapabilitiesV2.from_dict(
            _plain_json_value(raw["capabilities"])
        )
    except (TypeError, ValueError) as exc:
        raise _AdapterConfigurationError(
            "adapter capabilities are invalid"
        ) from exc
    if (
        capabilities.adapter_id != SUBPROCESS_JSON_ADAPTER_KIND
        or capabilities.adapter_version != SUBPROCESS_JSON_ADAPTER_VERSION
        or capabilities.subprocess_wire_version != SUBPROCESS_WIRE_VERSION
        or capabilities.input_schema_version != EVAL_INPUT_SCHEMA_VERSION
        or capabilities.submission_schema_version
        != EVAL_SUBMISSION_SCHEMA_VERSION
        or capabilities.clarification_protocol
        != SUBPROCESS_CLARIFICATION_PROTOCOL
        or capabilities.trace_protocol != SUBPROCESS_TRACE_PROTOCOL
        or capabilities.isolation_profile != SUBPROCESS_ISOLATION_PROFILE
    ):
        raise _AdapterConfigurationError(
            "adapter capabilities do not identify subprocess-json-v2"
        )

    # AgentConfigSnapshot is recursively frozen.  Rechecking the digest keeps
    # launch configuration coupled to the exact snapshot parsed above.
    if snapshot.digest() != snapshot_digest:
        raise _AdapterConfigurationError("agent snapshot changed during validation")
    return _SubprocessConfiguration(
        command=tuple(command),
        environment_allowlist=tuple(environment),
        capabilities=capabilities,
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


def _read_current_posix_identity(
    expected: _PosixProcessIdentity,
) -> Optional[_PosixProcessIdentity]:
    if not expected.birth_marker.startswith("proc:"):
        # The ps fallback has already returned this identity in the immediately
        # preceding bounded snapshot. Re-running ps once per descendant would
        # make cleanup time scale quadratically without closing its TOCTOU race.
        return expected
    try:
        with (Path("/proc") / str(expected.pid) / "stat").open("rb") as stream:
            return _parse_linux_process_stat(
                stream.read(_MAX_PROC_STAT_BYTES + 1)
            )
    except OSError:
        return None


def _signal_posix_identity(
    identity: _PosixProcessIdentity,
    signal_number: signal.Signals,
) -> bool:
    """Signal only the process generation observed by the tracker."""

    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if callable(pidfd_open) and callable(pidfd_send_signal):
        descriptor: Optional[int] = None
        try:
            descriptor = pidfd_open(identity.pid, 0)
            current = _read_current_posix_identity(identity)
            if (
                current is None
                or current.birth_marker != identity.birth_marker
                or current.terminated
            ):
                return True
            pidfd_send_signal(descriptor, signal_number, None, 0)
            return True
        except ProcessLookupError:
            return True
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return True
            if exc.errno not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP}:
                return False
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    current = _read_current_posix_identity(identity)
    if (
        current is None
        or current.birth_marker != identity.birth_marker
        or current.terminated
    ):
        return True
    try:
        os.kill(identity.pid, signal_number)
    except ProcessLookupError:
        return True
    except OSError as exc:
        return exc.errno == errno.ESRCH
    return True


def _signal_posix_group(process_group: int, signal_number: signal.Signals) -> bool:
    try:
        os.killpg(process_group, signal_number)
    except ProcessLookupError:
        return True
    except OSError as exc:
        return exc.errno == errno.ESRCH
    return True


def _freeze_posix_process_tree(
    process: subprocess.Popen[bytes],
    tracker: Optional[_PosixDescendantTracker],
) -> bool:
    successful = _signal_posix_group(process.pid, signal.SIGSTOP)
    if tracker is None:
        return successful

    previously_seen: set[Tuple[int, str]] = set()
    stable_rounds = 0
    for _ in range(_DESCENDANT_FREEZE_ROUNDS):
        identities = tracker.observe()
        current = {(identity.pid, identity.birth_marker) for identity in identities}
        for identity in identities:
            successful = (
                _signal_posix_identity(identity, signal.SIGSTOP) and successful
            )
        if current.issubset(previously_seen):
            stable_rounds += 1
        else:
            stable_rounds = 0
            previously_seen.update(current)
        if stable_rounds >= 1:
            break
        time.sleep(_POLL_SECONDS)
    return successful and tracker.complete


def _kill_tracked_posix_descendants(
    tracker: Optional[_PosixDescendantTracker],
) -> bool:
    if tracker is None:
        return True
    successful = True
    deadline = time.monotonic() + _DESCENDANT_REAP_SECONDS
    while True:
        identities = tracker.observe()
        if not identities:
            return successful and tracker.complete
        for identity in identities:
            successful = (
                _signal_posix_identity(identity, signal.SIGKILL) and successful
            )
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SECONDS)


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    job: _WindowsProcessJob,
    tracker: Optional[_PosixDescendantTracker] = None,
) -> bool:
    """Terminate the isolated Job/session plus known detached descendants."""

    successful = True
    if os.name == "nt":
        try:
            job.terminate()
        except Exception:
            successful = False
    else:
        successful = _freeze_posix_process_tree(process, tracker)
        if tracker is not None:
            for identity in tracker.observe():
                successful = (
                    _signal_posix_identity(identity, signal.SIGKILL) and successful
                )
        successful = (
            _signal_posix_group(process.pid, signal.SIGKILL) and successful
        )

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
    if os.name != "nt":
        successful = _kill_tracked_posix_descendants(tracker) and successful
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
    cancel_event: Optional[threading.Event] = None,
) -> BoundedProcessResult:
    process: Optional[subprocess.Popen[bytes]] = None
    job: Optional[_WindowsProcessJob] = None
    descendant_tracker: Optional[_PosixDescendantTracker] = None
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
        if os.name != "nt":
            descendant_tracker = _PosixDescendantTracker(process.pid)
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
        next_descendant_scan = 0.0
        while True:
            now = time.monotonic()
            if descendant_tracker is not None and now >= next_descendant_scan:
                descendant_tracker.observe()
                next_descendant_scan = now + _DESCENDANT_SCAN_SECONDS
            if cancel_event is not None and cancel_event.is_set():
                terminal_failure = FailureCode.PROCESS_KILLED
                break
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
            cleanup_ok = _terminate_process_tree(
                process,
                job,
                descendant_tracker,
            )
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
    target_materialization_id: str,
    code: FailureCode,
    elapsed_seconds: float,
    retryable: bool,
) -> EvalSubmission:
    return failure_submission(
        eval_input=eval_input,
        config=config,
        target_materialization_id=target_materialization_id,
        code=code,
        message=_FAILURE_MESSAGES[code],
        retryable=retryable,
        usage=empty_usage(elapsed_seconds=max(0.0, elapsed_seconds)),
    )


def _subprocess_invocation(
    *,
    eval_input: EvalInput,
    config: AgentRunConfig,
    target_access: TargetAccess,
    target_materialization_id: str,
) -> Dict[str, Any]:
    if target_access.target_materialization_id != target_materialization_id:
        raise _AdapterConfigurationError(
            "TargetAccess does not match materialization identity"
        )
    return {
        "schema_version": SUBPROCESS_INVOCATION_SCHEMA_VERSION,
        "eval_input": eval_input.to_dict(),
        "trial_binding": {
            "run_id": config.run_id,
            "task_id": config.task_id,
            "trial_id": config.trial_id,
            "trial_index": config.trial_index,
            "eval_input_digest": config.eval_input_digest,
            "wire_contract": config.wire_contract.to_dict(),
            "adapter_capabilities_digest": (
                config.adapter_capabilities_digest
            ),
        },
        "target_access": target_access.to_dict(),
        "materialization_id": target_materialization_id,
    }


class SubprocessAgentAdapter:
    """Run a snapshot-bound ``subprocess-json-v2`` Agent exactly once."""

    ADAPTER_KIND = SUBPROCESS_JSON_ADAPTER_KIND
    ADAPTER_VERSION = SUBPROCESS_JSON_ADAPTER_VERSION

    @staticmethod
    def compatibility(
        eval_input: EvalInput,
        config: AgentRunConfig,
    ) -> AdapterCompatibility:
        if not isinstance(eval_input, EvalInput) or not isinstance(
            config, AgentRunConfig
        ):
            raise TypeError("adapter compatibility requires canonical input/config")
        adapter = _adapter_configuration(config)
        if adapter.capabilities != config.adapter_capabilities:
            raise AgentAdapterIncompatibleError(
                AdapterIncompatibilityReason.CAPABILITY_MISMATCH
            )
        if (
            eval_input.review_target.kind is not config.target_kind
            or eval_input.review_target.kind
            not in adapter.capabilities.target_kinds
        ):
            raise AgentAdapterIncompatibleError(
                AdapterIncompatibilityReason.TARGET_KIND
            )
        return AdapterCompatibility()

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
        del clarification_channel  # Generic v2 is one-shot and has no channel wire.
        started = time.monotonic()

        if (
            not isinstance(eval_input, EvalInput)
            or not isinstance(config, AgentRunConfig)
            or eval_input.task_id != config.task_id
            or eval_input.digest() != config.eval_input_digest
        ):
            raise AgentAdapterError(
                FailureCode.SCHEMA_MISMATCH,
                "Adapter invocation input does not match its Trial binding",
                retryable=False,
            )

        try:
            if not isinstance(target_access, TargetAccess):
                raise _AdapterConfigurationError("TargetAccess type is invalid")
            if not isinstance(workspace, Path):
                raise _AdapterConfigurationError("workspace type is invalid")
            resolved_workspace = workspace.resolve(strict=True)
            if not resolved_workspace.is_dir():
                raise _AdapterConfigurationError("workspace is not a directory")
            adapter = _adapter_configuration(config)
            if adapter.capabilities != config.adapter_capabilities:
                raise AgentAdapterIncompatibleError(
                    AdapterIncompatibilityReason.CAPABILITY_MISMATCH
                )
            if not self.compatibility(eval_input, config).compatible:
                raise AgentAdapterIncompatibleError(
                    AdapterIncompatibilityReason.TARGET_KIND
                )
            argv = _expand_command(
                adapter,
                config=config,
                workspace=resolved_workspace,
            )
            environment = build_subprocess_environment(
                adapter.environment_allowlist
            )
            stdin_bytes = canonical_json_bytes(
                _subprocess_invocation(
                    eval_input=eval_input,
                    config=config,
                    target_access=target_access,
                    target_materialization_id=target_materialization_id,
                )
            )
        except AgentAdapterIncompatibleError:
            raise
        except _AdapterConfigurationError:
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
                code=FailureCode.ADAPTER_ERROR,
                elapsed_seconds=time.monotonic() - started,
                retryable=False,
            )
        except Exception:
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
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
                cancel_event=cancel_event,
            )
        except Exception:
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
                code=FailureCode.ADAPTER_ERROR,
                elapsed_seconds=time.monotonic() - started,
                retryable=True,
            )
        elapsed = time.monotonic() - started
        if result.failure_code is not None:
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
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
                target_materialization_id=target_materialization_id,
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
                target_materialization_id=target_materialization_id,
                code=code,
                elapsed_seconds=elapsed,
                retryable=code is FailureCode.PROCESS_KILLED,
            )

        try:
            submission = parse_submission_output(
                result.stdout,
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
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
                target_materialization_id=target_materialization_id,
                code=code,
                elapsed_seconds=time.monotonic() - started,
                retryable=False,
            )
        except Exception:
            return _failed_submission(
                eval_input=eval_input,
                config=config,
                target_materialization_id=target_materialization_id,
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
    "SUBPROCESS_JSON_ADAPTER_VERSION",
    "SUBPROCESS_INVOCATION_SCHEMA_VERSION",
    "SUBPROCESS_WIRE_VERSION",
    "SubprocessAgentAdapter",
    "build_subprocess_environment",
    "returncode_was_killed",
    "run_bounded_process",
    "subprocess_adapter_capabilities",
]
