from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tempfile
from threading import Event, Thread
import time
from typing import Any, Callable

from review_agent.models import QualityGateResult
from review_agent.quality import (
    QUALITY_GATE_STATUSES,
    MAX_REVISION_BLOB_BYTES,
    MAX_REVISION_FILES,
    QualityGateDefinition,
    QualityGateExecution,
    _read_blobs,
    _tree_blobs_at_revision,
    run_python_compile_gate,
)


DEFAULT_MAX_OUTPUT_BYTES = 1_048_576
MAX_SNAPSHOT_FILES = MAX_REVISION_FILES
SANDBOX_PROFILE = "revision_snapshot:no_shell:minimal_env:python_network_denied"

_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)\b"
        r"(\s*[:=]\s*)([^\s,;]+)"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
)

_PYTHON_NETWORK_GUARD = """\
import socket

_ReviewAgentOriginalSocket = socket.socket

class _ReviewAgentDeniedSocket(_ReviewAgentOriginalSocket):
    def __new__(cls, *args, **kwargs):
        raise PermissionError("network access denied by review-agent Quality Gate sandbox")

def _review_agent_network_denied(*args, **kwargs):
    raise PermissionError("network access denied by review-agent Quality Gate sandbox")

socket.socket = _ReviewAgentDeniedSocket
socket.create_connection = _review_agent_network_denied
socket.create_server = _review_agent_network_denied
socket.socketpair = _review_agent_network_denied
"""

_PYTHON_GATE_BOOTSTRAP = """\
import runpy
import socket
import sys

_ReviewAgentOriginalSocket = socket.socket

class _ReviewAgentDeniedSocket(_ReviewAgentOriginalSocket):
    def __new__(cls, *args, **kwargs):
        raise PermissionError("network access denied by review-agent Quality Gate sandbox")

def _review_agent_network_denied(*args, **kwargs):
    raise PermissionError("network access denied by review-agent Quality Gate sandbox")

socket.socket = _ReviewAgentDeniedSocket
socket.create_connection = _review_agent_network_denied
socket.create_server = _review_agent_network_denied
socket.socketpair = _review_agent_network_denied
module = sys.argv[1]
sys.argv = [module, *sys.argv[2:]]
runpy.run_module(module, run_name="__main__", alter_sys=True)
"""


@dataclass(frozen=True)
class _ProcessResult:
    status: str
    exit_code: int | None
    duration_seconds: float
    output: str
    output_truncated: bool
    reason: str | None


def execute_quality_gate(
    repo_path: Path,
    revision: str,
    gate: QualityGateDefinition,
    *,
    module_available: Callable[[str], bool] | None = None,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> QualityGateExecution:
    if gate.name == "python_compile":
        result = run_python_compile_gate(repo_path, revision=revision)
        return QualityGateExecution(result=result, raw_output=result.summary)

    module = gate.command[2]
    checker = module_available or _module_available
    if not checker(module):
        reason = f"Python module is not installed in the local gate runtime: {module}"
        return QualityGateExecution(
            result=_terminal_result(
                gate,
                status="unavailable",
                summary=f"{gate.name} was not run: {reason}",
                reason=reason,
            ),
            raw_output=reason,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="review-agent-quality-") as raw_root:
            root = Path(raw_root)
            snapshot = root / "snapshot"
            guard = root / "guard"
            snapshot.mkdir()
            guard.mkdir()
            warnings = _materialize_revision(Path(repo_path), revision, snapshot)
            (guard / "sitecustomize.py").write_text(
                _PYTHON_NETWORK_GUARD,
                encoding="utf-8",
            )
            command = [
                sys.executable,
                "-I",
                "-c",
                _PYTHON_GATE_BOOTSTRAP,
                module,
                *gate.command[3:],
            ]
            process = _run_bounded_process(
                command,
                cwd=snapshot,
                env=_minimal_gate_environment(snapshot, guard),
                timeout_seconds=gate.timeout_seconds,
                max_output_bytes=max_output_bytes,
                output_path=root / "gate-output.bin",
            )
            warning_text = "".join(f"snapshot warning: {item}\n" for item in warnings)
            raw_output = _redact_output(warning_text + process.output)
            return QualityGateExecution(
                result=_terminal_result(
                    gate,
                    status=process.status,
                    summary=_process_summary(gate, process),
                    reason=process.reason,
                    exit_code=process.exit_code,
                    duration_seconds=process.duration_seconds,
                    output_truncated=process.output_truncated,
                ),
                raw_output=raw_output,
            )
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        return QualityGateExecution(
            result=_terminal_result(
                gate,
                status="error",
                summary=f"{gate.name} runner error: {reason}",
                reason=reason,
            ),
            raw_output=_redact_output(reason),
        )


def skipped_quality_gate_execution(
    gate: QualityGateDefinition,
    reason: str,
) -> QualityGateExecution:
    return QualityGateExecution(
        result=_terminal_result(
            gate,
            status="skipped",
            summary=f"{gate.name} skipped by Runtime policy: {reason}",
            reason=reason,
        ),
        raw_output=reason,
    )


def _terminal_result(
    gate: QualityGateDefinition,
    *,
    status: str,
    summary: str,
    reason: str | None = None,
    exit_code: int | None = None,
    duration_seconds: float = 0.0,
    output_truncated: bool = False,
) -> QualityGateResult:
    if status not in QUALITY_GATE_STATUSES:
        raise ValueError("unsupported Quality Gate terminal status")
    return QualityGateResult(
        name=gate.name,
        status=status,
        command=list(gate.command),
        summary=summary,
        category=gate.category,
        cost=gate.cost,
        source=gate.source,
        blocking=gate.blocking,
        reason=reason,
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        output_truncated=output_truncated,
        sandbox=("git_blob_compile" if gate.name == "python_compile" else SANDBOX_PROFILE),
    )


def _process_summary(gate: QualityGateDefinition, process: _ProcessResult) -> str:
    if process.status == "passed":
        return f"{gate.name} passed in {process.duration_seconds:.2f}s"
    if process.status == "failed":
        return (
            f"{gate.name} failed with exit code {process.exit_code} "
            f"after {process.duration_seconds:.2f}s"
        )
    return f"{gate.name} {process.status}: {process.reason or 'unknown reason'}"


def _run_bounded_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    max_output_bytes: int,
    output_path: Path,
) -> _ProcessResult:
    if type(max_output_bytes) is not int or max_output_bytes < 1:
        raise ValueError("max_output_bytes must be a positive integer")
    started = time.monotonic()
    timed_out = False
    output_limited = False
    creationflags = 0
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        creationflags=creationflags,
        **popen_kwargs,
    )
    if process.stdout is None:
        process.kill()
        process.wait(timeout=5)
        raise RuntimeError("unable to capture Quality Gate output")

    captured = bytearray()
    output_limit_reached = Event()
    reader_finished = Event()
    reader_errors: list[Exception] = []

    def read_output() -> None:
        try:
            while True:
                chunk = process.stdout.read1(65_536)
                if not chunk:
                    return
                remaining = max_output_bytes - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    output_limit_reached.set()
                    return
        except Exception as error:  # pragma: no cover - defensive pipe failure.
            reader_errors.append(error)
        finally:
            reader_finished.set()

    reader = Thread(
        target=read_output,
        name="quality-gate-output-reader",
        daemon=True,
    )
    reader.start()
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            timed_out = True
            _terminate_process_tree(process)
            break
        if output_limit_reached.is_set():
            output_limited = True
            _terminate_process_tree(process)
            break
        output_limit_reached.wait(min(0.02, timeout_seconds - elapsed))
    if output_limit_reached.is_set():
        output_limited = True
        if process.poll() is None:
            _terminate_process_tree(process)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    if not reader_finished.wait(timeout=5):
        process.stdout.close()
        reader_finished.wait(timeout=1)
    reader.join(timeout=0)
    process.stdout.close()
    if output_limit_reached.is_set():
        output_limited = True

    elapsed = time.monotonic() - started
    raw = bytes(captured)
    output_path.write_bytes(raw)
    truncated = output_limited
    output = raw.decode("utf-8", errors="replace")
    if reader_errors:
        return _ProcessResult(
            status="error",
            exit_code=None,
            duration_seconds=elapsed,
            output=output,
            output_truncated=truncated,
            reason=f"output capture failed: {reader_errors[0]}",
        )
    if not reader_finished.is_set():
        return _ProcessResult(
            status="error",
            exit_code=None,
            duration_seconds=elapsed,
            output=output,
            output_truncated=truncated,
            reason="output capture did not terminate",
        )
    if timed_out:
        return _ProcessResult(
            status="timed_out",
            exit_code=None,
            duration_seconds=elapsed,
            output=output,
            output_truncated=truncated,
            reason=f"wall-clock timeout exceeded ({timeout_seconds:.2f}s)",
        )
    if output_limited:
        return _ProcessResult(
            status="error",
            exit_code=None,
            duration_seconds=elapsed,
            output=output,
            output_truncated=True,
            reason=f"output limit exceeded ({max_output_bytes} bytes)",
        )
    return _ProcessResult(
        status="passed" if process.returncode == 0 else "failed",
        exit_code=process.returncode,
        duration_seconds=elapsed,
        output=output,
        output_truncated=truncated,
        reason=None,
    )


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            terminated = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            if terminated.returncode == 0:
                return
            process.kill()
            return
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        process.kill()


def _minimal_gate_environment(snapshot: Path, guard: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for name in ("PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    home = guard.parent / "home"
    temp = guard.parent / "tmp"
    home.mkdir()
    temp.mkdir()
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "TEMP": str(temp),
            "TMP": str(temp),
            "PYTHONPATH": os.pathsep.join((str(guard), str(snapshot))),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_COLOR": "1",
            "CI": "1",
            "HTTP_PROXY": "http://127.0.0.1:1",
            "HTTPS_PROXY": "http://127.0.0.1:1",
            "ALL_PROXY": "http://127.0.0.1:1",
            "NO_PROXY": "",
        }
    )
    return env


def _materialize_revision(repo: Path, revision: str, destination: Path) -> list[str]:
    blobs = _tree_blobs_at_revision(repo, revision)
    if len(blobs) > MAX_SNAPSHOT_FILES:
        raise RuntimeError(
            f"revision snapshot exceeds file limit ({len(blobs)} > {MAX_SNAPSHOT_FILES})"
        )
    warnings: list[str] = []
    regular_blobs = [
        blob for blob in blobs if blob.mode in {"100644", "100755"}
    ]
    contents = _read_blobs(
        repo,
        [blob.object_id for blob in regular_blobs],
        max_total_bytes=MAX_REVISION_BLOB_BYTES,
    )
    total_bytes = 0
    root = destination.resolve()
    for blob in blobs:
        if blob.mode not in {"100644", "100755"}:
            warnings.append(f"skipped unsupported Git mode {blob.mode}: {blob.path}")
            continue
        relative = _safe_snapshot_path(blob.path)
        raw = contents[blob.object_id]
        total_bytes += len(raw)
        if total_bytes > MAX_REVISION_BLOB_BYTES:
            raise RuntimeError(
                "revision snapshot exceeds byte limit "
                f"({total_bytes} > {MAX_REVISION_BLOB_BYTES})"
            )
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved = target.resolve()
        if root != resolved and root not in resolved.parents:
            raise RuntimeError(f"unsafe revision snapshot path: {blob.path}")
        target.write_bytes(raw)
        if blob.mode == "100755" and os.name != "nt":
            target.chmod(target.stat().st_mode | 0o100)
    return warnings


def _safe_snapshot_path(path: str) -> PurePosixPath:
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        raise RuntimeError("revision contains an unsafe snapshot path")
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise RuntimeError(f"revision contains an unsafe snapshot path: {path}")
    if any(part.endswith((" ", ".")) or ":" in part for part in relative.parts):
        raise RuntimeError(f"revision contains a non-portable snapshot path: {path}")
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
    if any(part.split(".", 1)[0].casefold() in reserved for part in relative.parts):
        raise RuntimeError(f"revision contains a reserved snapshot path: {path}")
    return relative


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _redact_output(output: str) -> str:
    redacted = output
    redacted = _SECRET_PATTERNS[0].sub(r"\1\2[REDACTED]", redacted)
    redacted = _SECRET_PATTERNS[1].sub("Bearer [REDACTED]", redacted)
    return redacted
