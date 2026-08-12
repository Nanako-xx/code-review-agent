from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Protocol


MAX_QUALITY_TIMEOUT_SECONDS = 1800.0
MAX_INLINE_OUTPUT_BYTES = 50_000
MAX_OUTPUT_PREVIEW_BYTES = 2_048


class QualityGateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class PreflightArtifactSink(Protocol):
    def publish(
        self,
        *,
        snapshot_id: str,
        logical_name: str,
        content: bytes,
        content_type: str,
    ) -> str:
        ...


class QualityCommandExecutor(Protocol):
    def run(
        self,
        repository: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
    ) -> "CommandExecution":
        ...


_QUALITY_KINDS = frozenset({"syntax", "compile", "type", "lint", "build"})
_NETWORK_EXECUTABLES = frozenset(
    {"curl", "wget", "npx", "uvx", "pip", "pip3"}
)
_PACKAGE_MANAGERS = frozenset(
    {"npm", "pnpm", "yarn", "poetry", "uv", "conda"}
)
_INSTALL_VERBS = frozenset({"add", "ci", "install", "sync", "update", "upgrade"})


def _executable_name(value: str) -> str:
    name = Path(value).name.casefold()
    for suffix in (".exe", ".cmd", ".bat"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


@dataclass(frozen=True)
class LocalQualityCommand:
    name: str
    kind: str
    argv: tuple[str, ...]
    timeout_seconds: float

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip() or self.name != self.name.strip():
            raise ValueError("quality command name must be canonical text")
        if self.kind not in _QUALITY_KINDS:
            raise ValueError("quality command kind must be syntax/compile/type/lint/build")
        if type(self.argv) is not tuple or not self.argv or any(
            type(item) is not str or not item or "\x00" in item for item in self.argv
        ):
            raise ValueError("quality command argv must be a non-empty string tuple")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or not 0 < float(self.timeout_seconds) <= MAX_QUALITY_TIMEOUT_SECONDS
        ):
            raise ValueError("quality command timeout must be between 0 and 1800 seconds")
        executable = _executable_name(self.argv[0])
        if executable in _NETWORK_EXECUTABLES:
            raise ValueError(
                "quality command cannot use network or download-on-demand executables"
            )
        if executable in _PACKAGE_MANAGERS and any(
            argument.casefold() in _INSTALL_VERBS for argument in self.argv[1:]
        ):
            raise ValueError("quality command cannot install or update dependencies")
        normalized_arguments = tuple(argument.casefold() for argument in self.argv[1:])
        if any(
            normalized_arguments[index : index + 2] == ("-m", module)
            and any(
                argument in _INSTALL_VERBS
                for argument in normalized_arguments[index + 2 :]
            )
            for module in ("pip", "pip3")
            for index in range(max(0, len(normalized_arguments) - 1))
        ):
            raise ValueError("quality command cannot install or download dependencies")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))


@dataclass(frozen=True)
class LocalQualityPlan:
    commands: tuple[LocalQualityCommand, ...]
    total_timeout_seconds: float = MAX_QUALITY_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if type(self.commands) is not tuple or any(
            type(command) is not LocalQualityCommand for command in self.commands
        ):
            raise ValueError("quality plan commands must be a command tuple")
        if len({command.name for command in self.commands}) != len(self.commands):
            raise ValueError("quality plan command names must be unique")
        if (
            isinstance(self.total_timeout_seconds, bool)
            or not isinstance(self.total_timeout_seconds, (int, float))
            or not math.isfinite(self.total_timeout_seconds)
            or not 0 < float(self.total_timeout_seconds) <= MAX_QUALITY_TIMEOUT_SECONDS
        ):
            raise ValueError("quality stage timeout must be between 0 and 1800 seconds")
        object.__setattr__(
            self,
            "total_timeout_seconds",
            float(self.total_timeout_seconds),
        )


@dataclass(frozen=True)
class CommandExecution:
    exit_code: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    duration_seconds: float = 0.0
    timed_out: bool = False
    unavailable: bool = False
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ValueError("command execution exit_code must be an integer or null")
        if type(self.stdout) is not bytes or type(self.stderr) is not bytes:
            raise ValueError("command execution output must be bytes")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ValueError("command execution duration is invalid")
        if type(self.timed_out) is not bool or type(self.unavailable) is not bool:
            raise ValueError("command execution flags must be booleans")
        if self.timed_out and self.unavailable:
            raise ValueError("command execution cannot timeout and be unavailable")
        if (self.timed_out or self.unavailable or self.error_code is not None) and (
            self.exit_code is not None
        ):
            raise ValueError("failed command execution must not claim an exit code")


@dataclass(frozen=True)
class QualityCommandResult:
    name: str
    kind: str
    status: QualityGateStatus
    exit_code: int | None
    duration_ms: int
    reason_code: str
    diagnostic_summary: str
    stdout_preview: str
    stderr_preview: str
    stdout_artifact_id: str | None
    stderr_artifact_id: str | None
    stdout_truncated: bool
    stderr_truncated: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "reason_code": self.reason_code,
            "diagnostic_summary": self.diagnostic_summary,
            "stdout_preview": self.stdout_preview,
            "stderr_preview": self.stderr_preview,
            "stdout_artifact_id": self.stdout_artifact_id,
            "stderr_artifact_id": self.stderr_artifact_id,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
        }


@dataclass(frozen=True)
class QualityGateResult:
    snapshot_id: str
    status: QualityGateStatus
    commands: tuple[QualityCommandResult, ...]
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.snapshot_id) is not str or re.fullmatch(
            r"S-[0-9a-f]{64}", self.snapshot_id
        ) is None:
            raise ValueError("quality gate snapshot_id is invalid")
        if not isinstance(self.status, QualityGateStatus):
            raise ValueError("quality gate status is invalid")
        if type(self.commands) is not tuple or any(
            type(command) is not QualityCommandResult for command in self.commands
        ):
            raise ValueError("quality gate commands are invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "quality_gate_result_v2",
            "snapshot_id": self.snapshot_id,
            "status": self.status.value,
            "commands": [command.to_dict() for command in self.commands],
            "reason_code": self.reason_code,
        }


class SubprocessQualityExecutor:
    def run(
        self,
        repository: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
    ) -> CommandExecution:
        started = time.monotonic()
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            not in {
                "ALL_PROXY",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "NO_PROXY",
            }
        }
        environment.update(
            {
                "CARGO_NET_OFFLINE": "true",
                "GIT_TERMINAL_PROMPT": "0",
                "NPM_CONFIG_OFFLINE": "true",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INDEX": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        try:
            result = subprocess.run(
                list(argv),
                cwd=repository,
                env=environment,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
        except FileNotFoundError:
            return CommandExecution(
                exit_code=None,
                unavailable=True,
                error_code="tool_not_found",
                duration_seconds=time.monotonic() - started,
            )
        except subprocess.TimeoutExpired as error:
            return CommandExecution(
                exit_code=None,
                stdout=bytes(error.stdout or b""),
                stderr=bytes(error.stderr or b""),
                timed_out=True,
                error_code="command_timeout",
                duration_seconds=time.monotonic() - started,
            )
        except OSError:
            return CommandExecution(
                exit_code=None,
                error_code="process_error",
                duration_seconds=time.monotonic() - started,
            )
        return CommandExecution(
            exit_code=result.returncode,
            stdout=bytes(result.stdout),
            stderr=bytes(result.stderr),
            duration_seconds=time.monotonic() - started,
        )


def _output_projection(
    content: bytes,
    *,
    snapshot_id: str,
    logical_name: str,
    sink: PreflightArtifactSink,
) -> tuple[str, str | None, bool]:
    if len(content) <= MAX_INLINE_OUTPUT_BYTES:
        return content.decode("utf-8", "replace"), None, False
    artifact_id = sink.publish(
        snapshot_id=snapshot_id,
        logical_name=logical_name,
        content=content,
        content_type="text/plain",
    )
    preview = content[:MAX_OUTPUT_PREVIEW_BYTES].decode("utf-8", "replace")
    while len(preview.encode("utf-8")) > MAX_OUTPUT_PREVIEW_BYTES:
        preview = preview[:-1]
    return preview, artifact_id, True


def _diagnostic_summary(stdout: bytes, stderr: bytes, reason_code: str) -> str:
    source = stderr if stderr.strip() else stdout
    if not source.strip():
        return reason_code
    text = " ".join(source.decode("utf-8", "replace").split())
    return text[:2000]


class LocalQualityRunner:
    def __init__(self, *, executor: QualityCommandExecutor | None = None) -> None:
        self._executor = executor or SubprocessQualityExecutor()

    def run(
        self,
        repository: Path,
        snapshot_id: str,
        plan: LocalQualityPlan,
        sink: PreflightArtifactSink,
    ) -> QualityGateResult:
        if not isinstance(plan, LocalQualityPlan):
            raise ValueError("quality plan is invalid")
        if not hasattr(sink, "publish"):
            raise ValueError("quality output Artifact sink is required")
        if not plan.commands:
            return QualityGateResult(
                snapshot_id=snapshot_id,
                status=QualityGateStatus.UNAVAILABLE,
                commands=(),
                reason_code="no_configured_checks",
            )

        consumed = 0.0
        results: list[QualityCommandResult] = []
        for command in plan.commands:
            remaining = max(0.0, plan.total_timeout_seconds - consumed)
            if remaining <= 0:
                execution = CommandExecution(
                    exit_code=None,
                    timed_out=True,
                    error_code="stage_timeout",
                    duration_seconds=0,
                )
            else:
                execution = self._executor.run(
                    Path(repository),
                    command.argv,
                    min(command.timeout_seconds, remaining),
                )
            consumed += execution.duration_seconds
            if execution.unavailable:
                status = QualityGateStatus.UNAVAILABLE
                reason_code = execution.error_code or "tool_unavailable"
            elif execution.timed_out or execution.error_code is not None:
                status = QualityGateStatus.ERROR
                reason_code = execution.error_code or "quality_runtime_error"
            elif execution.exit_code == 0:
                status = QualityGateStatus.PASSED
                reason_code = "static_check_passed"
            else:
                status = QualityGateStatus.FAILED
                reason_code = "static_check_failed"
            stdout_preview, stdout_ref, stdout_truncated = _output_projection(
                execution.stdout,
                snapshot_id=snapshot_id,
                logical_name=f"quality/{command.name}/stdout.log",
                sink=sink,
            )
            stderr_preview, stderr_ref, stderr_truncated = _output_projection(
                execution.stderr,
                snapshot_id=snapshot_id,
                logical_name=f"quality/{command.name}/stderr.log",
                sink=sink,
            )
            results.append(
                QualityCommandResult(
                    name=command.name,
                    kind=command.kind,
                    status=status,
                    exit_code=execution.exit_code,
                    duration_ms=max(0, int(round(execution.duration_seconds * 1000))),
                    reason_code=reason_code,
                    diagnostic_summary=_diagnostic_summary(
                        execution.stdout,
                        execution.stderr,
                        reason_code,
                    ),
                    stdout_preview=stdout_preview,
                    stderr_preview=stderr_preview,
                    stdout_artifact_id=stdout_ref,
                    stderr_artifact_id=stderr_ref,
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
                )
            )

        statuses = {result.status for result in results}
        if QualityGateStatus.ERROR in statuses:
            aggregate = QualityGateStatus.ERROR
        elif QualityGateStatus.FAILED in statuses:
            aggregate = QualityGateStatus.FAILED
        elif QualityGateStatus.UNAVAILABLE in statuses:
            aggregate = QualityGateStatus.UNAVAILABLE
        else:
            aggregate = QualityGateStatus.PASSED
        return QualityGateResult(
            snapshot_id=snapshot_id,
            status=aggregate,
            commands=tuple(results),
        )


__all__ = [
    "CommandExecution",
    "LocalQualityCommand",
    "LocalQualityPlan",
    "LocalQualityRunner",
    "PreflightArtifactSink",
    "QualityCommandResult",
    "QualityGateResult",
    "QualityGateStatus",
    "SubprocessQualityExecutor",
]
