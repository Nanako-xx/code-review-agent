from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Iterable, Mapping, Protocol

from review_agent.tool_artifacts import ToolArtifactError, ToolResultArtifactStore
from review_agent.tool_result_protocol import ReviewToolResult, ToolErrorEnvelope


DEFAULT_REVIEW_TOOL_TIMEOUT_SECONDS = 300.0
_SNAPSHOT_ID = re.compile(r"\AS-[0-9a-f]{64}\Z")


class ReviewToolGatewayError(ValueError):
    pass


class ReviewToolFailure(Exception):
    def __init__(
        self,
        *,
        code: str,
        retryable: bool,
        message: str,
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.message = message
        self.exit_code = exit_code


@dataclass(frozen=True)
class ToolBackendResult:
    content: str
    reacquirable: bool
    exit_code: int | None = None

    def __post_init__(self) -> None:
        if type(self.content) is not str:
            raise ReviewToolGatewayError("Tool backend content must be text")
        try:
            self.content.encode("utf-8", "strict")
        except UnicodeError as error:
            raise ReviewToolGatewayError(
                "Tool backend content must be UTF-8"
            ) from error
        if type(self.reacquirable) is not bool:
            raise ReviewToolGatewayError(
                "Tool backend must declare call-level reacquirability"
            )
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ReviewToolGatewayError("Tool backend exit_code is invalid")


class ReviewToolBackend(Protocol):
    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        timeout_seconds: float,
    ) -> ToolBackendResult:
        ...


class ReviewToolGateway:
    """Execute read-only calls without ObservationStore or evidence IDs."""

    def __init__(
        self,
        *,
        snapshot_id: str,
        session_id: str,
        allowed_tools: Iterable[str],
        backend: ReviewToolBackend,
        timeout_seconds: float = DEFAULT_REVIEW_TOOL_TIMEOUT_SECONDS,
        artifact_store: ToolResultArtifactStore | None = None,
    ) -> None:
        if type(snapshot_id) is not str or _SNAPSHOT_ID.fullmatch(snapshot_id) is None:
            raise ReviewToolGatewayError("snapshot_id is invalid")
        if type(session_id) is not str or not session_id.strip():
            raise ReviewToolGatewayError("session_id must be non-empty")
        names = tuple(allowed_tools)
        if any(type(name) is not str or not name for name in names):
            raise ReviewToolGatewayError("allowed_tools contains an invalid name")
        if len(names) != len(set(names)):
            raise ReviewToolGatewayError("allowed_tools contains duplicates")
        if not hasattr(backend, "execute"):
            raise ReviewToolGatewayError("backend must implement execute")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ReviewToolGatewayError("timeout_seconds must be positive")
        if artifact_store is not None and not isinstance(
            artifact_store, ToolResultArtifactStore
        ):
            raise ReviewToolGatewayError(
                "artifact_store must be ToolResultArtifactStore or null"
            )
        if (
            artifact_store is not None
            and artifact_store.snapshot.snapshot_id != snapshot_id
        ):
            raise ReviewToolGatewayError(
                "Artifact Store Snapshot binding does not match"
            )
        self.snapshot_id = snapshot_id
        self.session_id = session_id
        self.allowed_tools = frozenset(names)
        self.backend = backend
        self.timeout_seconds = float(timeout_seconds)
        self.artifact_store = artifact_store

    def execute(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ReviewToolResult:
        if type(tool_call_id) is not str or not tool_call_id.strip():
            raise ReviewToolGatewayError("tool_call_id must be non-empty")
        if tool_name not in self.allowed_tools:
            return self._failure(
                tool_call_id,
                tool_name,
                arguments,
                ToolErrorEnvelope(
                    code="unsupported_operation",
                    retryable=False,
                    message="The Tool is not authorized for this Assignment",
                ),
            )
        if not isinstance(arguments, Mapping):
            return self._failure(
                tool_call_id,
                tool_name,
                {},
                ToolErrorEnvelope(
                    code="invalid_arguments",
                    retryable=False,
                    message="Tool arguments must be a JSON object",
                ),
            )
        try:
            normalized_arguments = json.loads(
                json.dumps(
                    dict(arguments),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError, UnicodeError):
            return self._failure(
                tool_call_id,
                tool_name,
                {},
                ToolErrorEnvelope(
                    code="invalid_arguments",
                    retryable=False,
                    message="Tool arguments must be canonical JSON",
                ),
            )

        path_value = normalized_arguments.get("path")
        if isinstance(path_value, str):
            if len(path_value) > 32_767:
                return self._failure(
                    tool_call_id,
                    tool_name,
                    normalized_arguments,
                    ToolErrorEnvelope(
                        code="path_too_long",
                        retryable=False,
                        message="The requested path exceeds the supported path length",
                    ),
                )
            path_parts = path_value.split("/")
            if (
                path_value.startswith(("/", "\\"))
                or "\\" in path_value
                or ":" in path_value
                or any(part in {"", ".", ".."} for part in path_parts)
            ):
                return self._failure(
                    tool_call_id,
                    tool_name,
                    normalized_arguments,
                    ToolErrorEnvelope(
                        code="unauthorized_path",
                        retryable=False,
                        message="The requested path is outside the authorized Snapshot",
                    ),
                )

        try:
            if tool_name == "read_artifact" and self.artifact_store is not None:
                try:
                    page = self.artifact_store.read_artifact(
                        normalized_arguments.get("artifact_id"),
                        cursor=normalized_arguments.get("cursor", 0),
                        max_chars=normalized_arguments.get("max_chars", 50_000),
                    )
                except ToolArtifactError as error:
                    code = (
                        "missing_artifact"
                        if "unavailable" in str(error).casefold()
                        else "invalid_arguments"
                    )
                    raise ReviewToolFailure(
                        code=code,
                        retryable=False,
                        message=(
                            "The requested Artifact is unavailable"
                            if code == "missing_artifact"
                            else "Artifact read arguments are invalid"
                        ),
                    ) from error
                backend_result = ToolBackendResult(
                    content=json.dumps(
                        {
                            "artifact_id": page.artifact_id,
                            "cursor": page.cursor,
                            "next_cursor": page.next_cursor,
                            "has_more": page.has_more,
                            "total_chars": page.total_chars,
                            "content": page.content,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    reacquirable=True,
                )
            else:
                backend_result = self.backend.execute(
                    tool_name,
                    normalized_arguments,
                    self.timeout_seconds,
                )
            if not isinstance(backend_result, ToolBackendResult):
                raise ReviewToolFailure(
                    code="invalid_tool_result",
                    retryable=False,
                    message="Tool backend returned an invalid result",
                )
            return ReviewToolResult.success(
                tool_call_id=tool_call_id,
                session_id=self.session_id,
                snapshot_id=self.snapshot_id,
                tool_name=tool_name,
                arguments=normalized_arguments,
                content=backend_result.content,
                reacquirable=backend_result.reacquirable,
                exit_code=backend_result.exit_code,
            )
        except ReviewToolFailure as error:
            envelope = ToolErrorEnvelope(
                code=error.code,
                retryable=error.retryable,
                message=error.message,
                exit_code=error.exit_code,
            )
        except TimeoutError:
            envelope = ToolErrorEnvelope(
                code="tool_timeout",
                retryable=True,
                message="Tool execution exceeded the configured timeout",
            )
        except ValueError:
            envelope = ToolErrorEnvelope(
                code="invalid_arguments",
                retryable=False,
                message="Tool arguments are invalid",
            )
        except OSError:
            envelope = ToolErrorEnvelope(
                code="transient_io",
                retryable=True,
                message="Tool execution encountered transient I/O",
            )
        except Exception:
            envelope = ToolErrorEnvelope(
                code="tool_error",
                retryable=False,
                message="Tool execution failed",
            )
        return self._failure(
            tool_call_id,
            tool_name,
            normalized_arguments,
            envelope,
        )

    def _failure(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        error: ToolErrorEnvelope,
    ) -> ReviewToolResult:
        return ReviewToolResult.failure(
            tool_call_id=tool_call_id,
            session_id=self.session_id,
            snapshot_id=self.snapshot_id,
            tool_name=tool_name,
            arguments=arguments,
            error=error,
        )


__all__ = [
    "DEFAULT_REVIEW_TOOL_TIMEOUT_SECONDS",
    "ReviewToolBackend",
    "ReviewToolFailure",
    "ReviewToolGateway",
    "ReviewToolGatewayError",
    "ToolBackendResult",
]
