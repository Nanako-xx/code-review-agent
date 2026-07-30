from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from collections.abc import Iterable
from typing import Any

from review_agent.context import remote_visible_memory_snapshot
from review_agent.memory_models import (
    DurableMemoryRecord,
    MemoryScope,
    MemorySnapshot,
)
from review_agent.memory_retrieval import (
    HardPolicyBudgetExceeded,
    MemoryQuery,
    MemoryRetrievalError,
    ProjectionBudgetExceeded,
    RetrievalLimits,
    SnapshotMemoryQueryService,
)
from review_agent.observations import ObservationStore
from review_agent.repository_intelligence import collect_python_symbols, search_repository_text


class ToolGatewayError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "tool_error",
        tool_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.tool_name = tool_name


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    observation_ids: list[str]
    context_view: str
    truncated: bool = False


SUPPORTED_TOOL_NAMES = (
    "read_range",
    "compare_base_head",
    "search_code",
    "list_symbols",
    "inspect_symbol",
    "find_references",
    "read_commit_messages",
    "query_project_memory",
)


class ToolGateway:
    def __init__(
        self,
        repository_path: Path,
        base_revision: str,
        head_revision: str,
        observation_store: ObservationStore,
        max_context_chars: int = 4000,
        timeout_seconds: int = 10,
        max_commit_messages: int = 50,
        max_commit_body_chars: int = 4000,
        allowed_tools: Iterable[str] | None = None,
        memory_query_service: SnapshotMemoryQueryService | None = None,
        snapshot_query_service: SnapshotMemoryQueryService | None = None,
        memory_snapshot: MemorySnapshot | None = None,
        assignment_id: str | None = None,
        assignment_scope: MemoryScope | None = None,
        memory_query_limits: RetrievalLimits | None = None,
    ) -> None:
        self.repository_path = Path(repository_path)
        self.base_revision = base_revision
        self.head_revision = head_revision
        self.observation_store = observation_store
        if type(max_context_chars) is not int or max_context_chars < 1:
            raise ValueError("max_context_chars must be a positive integer")
        self.max_context_chars = max_context_chars
        self.timeout_seconds = timeout_seconds
        if type(max_commit_messages) is not int or max_commit_messages < 1:
            raise ValueError("max_commit_messages must be a positive integer")
        if type(max_commit_body_chars) is not int or max_commit_body_chars < 1:
            raise ValueError("max_commit_body_chars must be a positive integer")
        self.max_commit_messages = max_commit_messages
        self.max_commit_body_chars = max_commit_body_chars
        if isinstance(allowed_tools, (str, bytes)):
            raise ValueError("allowed_tools must be an iterable of tool names")
        requested_tools = (
            tuple(SUPPORTED_TOOL_NAMES)
            if allowed_tools is None
            else tuple(allowed_tools)
        )
        if any(not isinstance(name, str) or not name for name in requested_tools):
            raise ValueError("allowed_tools must contain non-empty strings")
        unsupported = set(requested_tools) - set(SUPPORTED_TOOL_NAMES)
        if unsupported:
            raise ValueError(
                "unsupported allowed tool(s): " + ", ".join(sorted(unsupported))
            )
        self.allowed_tools = frozenset(requested_tools)
        if (
            memory_query_service is not None
            and snapshot_query_service is not None
            and memory_query_service is not snapshot_query_service
        ):
            raise ValueError(
                "memory_query_service and snapshot_query_service conflict"
            )
        if assignment_id is not None and (
            not isinstance(assignment_id, str) or not assignment_id.strip()
        ):
            raise ValueError("assignment_id must be a non-empty string")
        if assignment_scope is not None and type(assignment_scope) is not MemoryScope:
            raise ValueError("assignment_scope must be a canonical MemoryScope")
        if memory_query_limits is not None and type(memory_query_limits) is not RetrievalLimits:
            raise ValueError("memory_query_limits must be canonical RetrievalLimits")
        query_service = (
            memory_query_service
            if memory_query_service is not None
            else snapshot_query_service
        )
        source_snapshot: MemorySnapshot | None = None
        bound_assignment_id: str | None = None
        bound_assignment_scope: MemoryScope | None = None
        query_limits: RetrievalLimits | None = None
        if memory_snapshot is not None:
            if query_service is not None:
                raise ValueError(
                    "provide either a MemorySnapshot or Snapshot query service, not both"
                )
            if type(memory_snapshot) is not MemorySnapshot:
                raise ValueError("memory_snapshot must be a canonical MemorySnapshot")
            if not isinstance(assignment_id, str) or not assignment_id.strip():
                raise ValueError(
                    "assignment_id is required when binding a MemorySnapshot"
                )
            if type(assignment_scope) is not MemoryScope:
                raise ValueError(
                    "assignment_scope is required when binding a MemorySnapshot"
                )
            source_snapshot = MemorySnapshot.from_dict(memory_snapshot.to_dict())
            bound_assignment_id = assignment_id
            bound_assignment_scope = MemoryScope.from_dict(assignment_scope.to_dict())
            query_limits = replace(memory_query_limits or RetrievalLimits())
        if query_service is not None and not isinstance(
            query_service,
            SnapshotMemoryQueryService,
        ):
            raise ValueError(
                "memory query source must be a SnapshotMemoryQueryService; live stores are forbidden"
            )
        if query_service is not None:
            service_snapshot = getattr(query_service, "_snapshot", None)
            service_assignment_id = getattr(query_service, "_assignment_id", None)
            service_assignment_scope = getattr(query_service, "_assignment_scope", None)
            service_limits = getattr(query_service, "limits", None)
            if type(service_snapshot) is not MemorySnapshot:
                raise ValueError("memory query service does not hold a canonical Snapshot")
            if not isinstance(service_assignment_id, str) or not service_assignment_id:
                raise ValueError("memory query service has no canonical Assignment ID")
            if type(service_assignment_scope) is not MemoryScope:
                raise ValueError("memory query service has no canonical Assignment scope")
            if type(service_limits) is not RetrievalLimits:
                raise ValueError("memory query service has no canonical retrieval limits")
            source_snapshot = MemorySnapshot.from_dict(service_snapshot.to_dict())
            bound_assignment_id = assignment_id or service_assignment_id
            bound_assignment_scope = (
                MemoryScope.from_dict(assignment_scope.to_dict())
                if assignment_scope is not None
                else MemoryScope.from_dict(service_assignment_scope.to_dict())
            )
            query_limits = replace(memory_query_limits or service_limits)
        elif memory_snapshot is None and any(
            value is not None
            for value in (assignment_id, assignment_scope, memory_query_limits)
        ):
            raise ValueError("assignment binding options require a MemorySnapshot")

        if query_limits is not None and type(query_limits) is not RetrievalLimits:
            raise ValueError("memory_query_limits must be canonical RetrievalLimits")

        verified_head_sha: str | None = None
        remote_snapshot: MemorySnapshot | None = None
        if source_snapshot is not None:
            verified_head_sha = _resolve_commit_sha(
                self.repository_path,
                self.head_revision,
                self.timeout_seconds,
            )
            if source_snapshot.head_sha.casefold() != verified_head_sha.casefold():
                raise ValueError(
                    "memory query Snapshot head does not match the reviewed head revision"
                )
            if bound_assignment_id is None or bound_assignment_scope is None:
                raise ValueError("memory query service is missing its Assignment binding")
            remote_snapshot = remote_visible_memory_snapshot(source_snapshot)
            query_service = SnapshotMemoryQueryService(
                remote_snapshot,
                assignment_id=bound_assignment_id,
                assignment_scope=bound_assignment_scope,
                limits=query_limits,
            )
        else:
            query_service = None

        self.verified_head_sha = verified_head_sha
        self.memory_snapshot = remote_snapshot
        self.memory_assignment_id = bound_assignment_id
        self.memory_assignment_scope = bound_assignment_scope
        self.memory_query_service = query_service
        self._memory_context_limit_bytes: int | None = None
        self._memory_context_used_bytes = 0
        self.attempted_tool_calls = 0
        self.denied_tool_calls = 0

    def bind_memory_context_ledger(
        self,
        *,
        limit_bytes: int,
        initial_bytes: int,
    ) -> None:
        """Bind the initial Memory projection and later tool results to one ledger."""

        if type(limit_bytes) is not int or limit_bytes < 0:
            raise ValueError("memory context ledger limit must be non-negative")
        if type(initial_bytes) is not int or not 0 <= initial_bytes <= limit_bytes:
            raise ValueError("initial Memory bytes exceed the context ledger")
        service = self.memory_query_service
        if service is None:
            if limit_bytes or initial_bytes:
                raise ValueError("memory context ledger requires a Snapshot service")
            return
        if service.call_count or self._memory_context_used_bytes:
            raise ValueError("memory context ledger must be bound before Memory queries")
        self._memory_context_limit_bytes = limit_bytes
        self._memory_context_used_bytes = initial_bytes

    @property
    def memory_context_limit_bytes(self) -> int | None:
        return self._memory_context_limit_bytes

    @property
    def memory_context_used_bytes(self) -> int:
        return self._memory_context_used_bytes

    @property
    def memory_context_remaining_bytes(self) -> int | None:
        if self._memory_context_limit_bytes is None:
            return None
        return max(
            0,
            self._memory_context_limit_bytes - self._memory_context_used_bytes,
        )

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        self.attempted_tool_calls += 1
        if tool_name not in self.allowed_tools:
            self.denied_tool_calls += 1
            raise ToolGatewayError(
                f"tool is not allowed for this reviewer task: {tool_name}",
                code="tool_not_allowed",
                tool_name=tool_name,
            )
        if tool_name == "read_range":
            return self._read_range(arguments)
        if tool_name == "compare_base_head":
            return self._compare_base_head(arguments)
        if tool_name == "search_code":
            return self._search_code(arguments)
        if tool_name == "list_symbols":
            return self._list_symbols(arguments)
        if tool_name == "inspect_symbol":
            return self._inspect_symbol(arguments)
        if tool_name == "find_references":
            return self._find_references(arguments)
        if tool_name == "read_commit_messages":
            return self._read_commit_messages(arguments)
        if tool_name == "query_project_memory":
            return self._query_project_memory(arguments)
        self.denied_tool_calls += 1
        raise ToolGatewayError(
            f"unsupported tool: {tool_name}",
            code="unsupported_tool",
            tool_name=tool_name,
        )

    def _read_range(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        path = _safe_repo_path(_require_non_empty_string(arguments, "path"))
        revision_label, revision = self._resolve_revision(str(arguments.get("revision", "head")))
        line_start = int(arguments["line_start"])
        line_end = int(arguments["line_end"])
        if line_start < 1 or line_end < line_start:
            raise ToolGatewayError("invalid line range")
        content = _git_show(self.repository_path, revision, path, self.timeout_seconds)
        lines = content.splitlines()
        selected = "\n".join(lines[line_start - 1 : line_end])
        if selected:
            selected += "\n"
        context_view, truncated = _context_view(selected, self.max_context_chars)
        observation = self.observation_store.record(
            source="git.read_range",
            revision=f"{revision_label}@{revision}",
            path=path,
            line_start=line_start,
            line_end=line_end,
            raw_content=selected,
            context_view=context_view,
        )
        return ToolExecutionResult("read_range", [observation.observation_id], context_view, truncated)

    def _compare_base_head(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        path = _safe_repo_path(_require_non_empty_string(arguments, "path"))
        raw = _run_git(
            self.repository_path,
            ["diff", "--unified=3", f"{self.base_revision}..{self.head_revision}", "--", path],
            self.timeout_seconds,
            allow_exit_codes={0},
        )
        context_view, truncated = _context_view(raw, self.max_context_chars)
        observation = self.observation_store.record(
            source="git.compare_base_head",
            revision=f"{self.base_revision}..{self.head_revision}",
            path=path,
            line_start=None,
            line_end=None,
            raw_content=raw,
            context_view=context_view,
        )
        return ToolExecutionResult("compare_base_head", [observation.observation_id], context_view, truncated)

    def _search_code(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        query = _require_non_empty_string(arguments, "query")
        revision_label, revision = self._resolve_revision(str(arguments.get("revision", "head")))
        max_results = int(arguments.get("max_results", 20))
        if max_results < 1:
            raise ToolGatewayError("max_results must be positive")
        raw = _run_git(
            self.repository_path,
            ["grep", "-n", "--fixed-strings", query, revision, "--", "."],
            self.timeout_seconds,
            allow_exit_codes={0, 1},
        )
        lines = raw.splitlines()[:max_results]
        limited = "\n".join(lines)
        if limited:
            limited += "\n"
        context_view, truncated = _context_view(limited, self.max_context_chars)
        observation = self.observation_store.record(
            source="git.search_code",
            revision=f"{revision_label}@{revision}",
            path=None,
            line_start=None,
            line_end=None,
            raw_content=limited,
            context_view=context_view,
        )
        return ToolExecutionResult("search_code", [observation.observation_id], context_view, truncated)

    def _list_symbols(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        path = _safe_repo_path(_require_non_empty_string(arguments, "path"))
        revision_label, revision = self._resolve_revision(str(arguments.get("revision", "head")))
        symbols = collect_python_symbols(self.repository_path, revision, paths=[path])
        raw_content = json.dumps([asdict(symbol) for symbol in symbols], ensure_ascii=False, indent=2)
        lines = [
            f"{symbol.kind} {symbol.qualified_name} {symbol.path}:{symbol.line_start}-{symbol.line_end}"
            for symbol in symbols
        ]
        context_view, truncated = _context_view("\n".join(lines) or "- No Python symbols found", self.max_context_chars)
        observation = self.observation_store.record(
            source="repo_intelligence.list_symbols",
            revision=f"{revision_label}@{revision}",
            path=path,
            line_start=None,
            line_end=None,
            raw_content=raw_content,
            context_view=context_view,
        )
        return ToolExecutionResult("list_symbols", [observation.observation_id], context_view, truncated)

    def _inspect_symbol(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        name = _require_non_empty_string(arguments, "name")
        revision_label, revision = self._resolve_revision(str(arguments.get("revision", "head")))
        symbols = collect_python_symbols(self.repository_path, revision)
        matches = [symbol for symbol in symbols if symbol.name == name or symbol.qualified_name == name]
        raw_content = json.dumps([asdict(symbol) for symbol in matches], ensure_ascii=False, indent=2)
        lines = []
        for symbol in matches:
            calls = ", ".join(symbol.calls) if symbol.calls else "none"
            lines.append(
                f"{symbol.kind} {symbol.qualified_name} {symbol.path}:{symbol.line_start}-{symbol.line_end}; "
                f"calls: {calls}"
            )
        context_view, truncated = _context_view("\n".join(lines) or f"- No symbol found: {name}", self.max_context_chars)
        observation = self.observation_store.record(
            source="repo_intelligence.inspect_symbol",
            revision=f"{revision_label}@{revision}",
            path=matches[0].path if matches else None,
            line_start=matches[0].line_start if matches else None,
            line_end=matches[0].line_end if matches else None,
            raw_content=raw_content,
            context_view=context_view,
        )
        return ToolExecutionResult("inspect_symbol", [observation.observation_id], context_view, truncated)

    def _find_references(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        name = _require_non_empty_string(arguments, "name")
        revision_label, revision = self._resolve_revision(str(arguments.get("revision", "head")))
        max_results = int(arguments.get("max_results", 20))
        if max_results < 1:
            raise ToolGatewayError("max_results must be positive")
        matches = search_repository_text(self.repository_path, revision, name, max_results=max_results)
        raw_content = json.dumps([asdict(match) for match in matches], ensure_ascii=False, indent=2)
        lines = [f"{match.path}:{match.line_number}:{match.line}" for match in matches]
        context_view, truncated = _context_view("\n".join(lines) or f"- No references found: {name}", self.max_context_chars)
        observation = self.observation_store.record(
            source="repo_intelligence.find_references",
            revision=f"{revision_label}@{revision}",
            path=None,
            line_start=None,
            line_end=None,
            raw_content=raw_content,
            context_view=context_view,
        )
        return ToolExecutionResult("find_references", [observation.observation_id], context_view, truncated)

    def _read_commit_messages(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        _validate_commit_message_arguments(
            arguments,
            self.base_revision,
            self.head_revision,
        )
        requested_max = arguments.get("max_commits", self.max_commit_messages)
        if type(requested_max) is not int or requested_max < 1:
            raise ToolGatewayError("max_commits must be a positive integer")
        limit = min(requested_max, self.max_commit_messages)
        revision_range = f"{self.base_revision}..{self.head_revision}"
        raw_log = _run_git(
            self.repository_path,
            [
                "log",
                f"--max-count={limit + 1}",
                "--format=%H%x00%s%x00%b%x00%x1e",
                revision_range,
            ],
            self.timeout_seconds,
            allow_exit_codes={0},
        )
        parsed = _parse_commit_messages(
            raw_log,
            max_body_chars=self.max_commit_body_chars,
        )
        commit_count_truncated = len(parsed) > limit
        commits = parsed[:limit]
        raw_content = json.dumps(commits, ensure_ascii=False, indent=2)
        context_view, context_truncated = _context_view(
            raw_content,
            self.max_context_chars,
        )
        observation = self.observation_store.record(
            source="git.read_commit_messages",
            revision=revision_range,
            path=None,
            line_start=None,
            line_end=None,
            raw_content=raw_content,
            context_view=context_view,
        )
        return ToolExecutionResult(
            "read_commit_messages",
            [observation.observation_id],
            context_view,
            commit_count_truncated or context_truncated,
        )

    def _resolve_revision(self, revision_label: str) -> tuple[str, str]:
        if revision_label == "base":
            return "base", self.base_revision
        if revision_label == "head":
            return "head", self.head_revision
        raise ToolGatewayError(f"unauthorized revision: {revision_label}")

    def _query_project_memory(
        self,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        service = self.memory_query_service
        if service is None:
            raise ToolGatewayError(
                "query_project_memory is unavailable without an Assignment-bound Snapshot",
                code="memory_snapshot_unavailable",
                tool_name="query_project_memory",
            )
        query = _memory_query_from_arguments(arguments)
        try:
            result = service.query(query)
        except HardPolicyBudgetExceeded:
            raise
        except (MemoryRetrievalError, TypeError, ValueError) as error:
            raise ToolGatewayError(
                f"invalid Snapshot memory query: {error}",
                code="invalid_memory_query",
                tool_name="query_project_memory",
            ) from error

        byte_limit = min(service.limits.max_query_bytes, self.max_context_chars)
        remaining = self.memory_context_remaining_bytes
        if remaining is not None:
            byte_limit = min(byte_limit, remaining)
        try:
            payload, raw_content, final_size = _fit_memory_tool_payload(
                assignment_id=result.assignment_id,
                snapshot_id=result.snapshot_id,
                call_index=result.call_index,
                records=result.records,
                omitted_memory_ids=result.omitted_memory_ids,
                max_bytes=byte_limit,
            )
        except HardPolicyBudgetExceeded:
            raise
        except ProjectionBudgetExceeded as error:
            raise ToolGatewayError(
                f"Snapshot memory query exceeds the remaining Context budget: {error}",
                code="memory_context_budget_exceeded",
                tool_name="query_project_memory",
            ) from error

        if len(raw_content.encode("utf-8")) != payload["byte_size"]:
            raise AssertionError("final Memory tool payload byte_size is inconsistent")
        if final_size > byte_limit:
            raise AssertionError("final Memory tool payload escaped its byte limit")
        if self._memory_context_limit_bytes is not None:
            self._memory_context_used_bytes += final_size
        observation = self.observation_store.record(
            source="memory.query_project_memory",
            revision=f"head@{self.verified_head_sha}",
            path=query.path,
            line_start=None,
            line_end=None,
            raw_content=raw_content,
            context_view=raw_content,
        )
        return ToolExecutionResult(
            "query_project_memory",
            [observation.observation_id],
            raw_content,
            False,
        )


def _require_non_empty_string(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolGatewayError(f"{name} must be a non-empty string")
    return value


def _safe_repo_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not normalized or pure.parts[0] in {".git", ".env"}:
        raise ToolGatewayError(f"unsafe repository path: {path}")
    return normalized


def _memory_query_from_arguments(arguments: dict[str, Any]) -> MemoryQuery:
    if type(arguments) is not dict:
        raise ToolGatewayError(
            "query_project_memory arguments must be an object",
            code="invalid_memory_query",
            tool_name="query_project_memory",
        )
    allowed = {"assignment_id", "path", "symbol", "contract", "query"}
    unsupported = set(arguments) - allowed
    if unsupported:
        raise ToolGatewayError(
            "query_project_memory received unsupported argument(s): "
            + ", ".join(sorted(str(item) for item in unsupported)),
            code="invalid_memory_query",
            tool_name="query_project_memory",
        )
    if "assignment_id" not in arguments:
        raise ToolGatewayError(
            "query_project_memory requires assignment_id",
            code="invalid_memory_query",
            tool_name="query_project_memory",
        )
    for name in allowed.intersection(arguments):
        if not isinstance(arguments[name], str):
            raise ToolGatewayError(
                f"query_project_memory {name} must be a string",
                code="invalid_memory_query",
                tool_name="query_project_memory",
            )
    try:
        return MemoryQuery(
            assignment_id=arguments["assignment_id"],
            path=arguments.get("path"),
            symbol=arguments.get("symbol"),
            contract=arguments.get("contract"),
            query_text=arguments.get("query", ""),
        )
    except (TypeError, ValueError) as error:
        raise ToolGatewayError(
            f"invalid query_project_memory arguments: {error}",
            code="invalid_memory_query",
            tool_name="query_project_memory",
        ) from error


def _fit_memory_tool_payload(
    *,
    assignment_id: str,
    snapshot_id: str,
    call_index: int,
    records: tuple[DurableMemoryRecord, ...],
    omitted_memory_ids: tuple[str, ...],
    max_bytes: int,
) -> tuple[dict[str, Any], str, int]:
    selected = list(records)
    omitted = list(dict.fromkeys(omitted_memory_ids))
    while True:
        payload, raw_content, byte_size = _serialize_memory_tool_payload(
            assignment_id=assignment_id,
            snapshot_id=snapshot_id,
            call_index=call_index,
            records=selected,
            omitted_memory_ids=omitted,
        )
        if byte_size <= max_bytes:
            return payload, raw_content, byte_size

        ordinary = [
            record for record in reversed(selected) if record.policy_effect is None
        ]
        if ordinary:
            dropped = ordinary[0]
            selected.remove(dropped)
            if dropped.memory_id not in omitted:
                omitted.append(dropped.memory_id)
            continue

        hard_policy_ids = [
            record.memory_id
            for record in selected
            if record.policy_effect is not None
        ]
        if hard_policy_ids:
            raise HardPolicyBudgetExceeded(
                boundary="query_tool",
                budget="utf8_bytes",
                limit=max_bytes,
                required=byte_size,
                memory_ids=hard_policy_ids,
            )
        raise ProjectionBudgetExceeded(
            boundary="query_tool",
            limit=max_bytes,
            required=byte_size,
        )


def _serialize_memory_tool_payload(
    *,
    assignment_id: str,
    snapshot_id: str,
    call_index: int,
    records: list[DurableMemoryRecord],
    omitted_memory_ids: list[str],
) -> tuple[dict[str, Any], str, int]:
    byte_size = 0
    for _ in range(16):
        payload = {
            "assignment_id": assignment_id,
            "snapshot_id": snapshot_id,
            "call_index": call_index,
            "byte_size": byte_size,
            "records": [record.to_dict() for record in records],
            "omitted_memory_ids": list(omitted_memory_ids),
        }
        raw_content = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        measured = len(raw_content.encode("utf-8"))
        if measured == byte_size:
            return payload, raw_content, measured
        byte_size = measured
    raise AssertionError("Memory payload byte_size did not converge")


def _is_full_git_object_id(value: str) -> bool:
    return re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", value) is not None


def _resolve_commit_sha(
    repo: Path,
    revision: str,
    timeout_seconds: int,
) -> str:
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or revision != revision.strip()
        or revision.startswith("-")
        or "\x00" in revision
    ):
        raise ValueError("head revision must be a canonical Git revision")
    resolved = _run_git(
        repo,
        ["rev-parse", "--verify", f"{revision}^{{commit}}"],
        timeout_seconds,
        allow_exit_codes={0},
    ).strip()
    if not _is_full_git_object_id(resolved):
        raise ValueError("head revision did not resolve to a commit SHA")
    return resolved.casefold()


def _git_show(repo: Path, revision: str, path: str, timeout_seconds: int) -> str:
    return _run_git(repo, ["show", f"{revision}:{path}"], timeout_seconds, allow_exit_codes={0})


def _run_git(repo: Path, args: list[str], timeout_seconds: int, allow_exit_codes: set[int]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
    )
    if result.returncode not in allow_exit_codes:
        raise ToolGatewayError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


_COMMIT_MESSAGE_ARGUMENTS = {
    "max_commits",
    "base_revision",
    "head_revision",
    "base",
    "head",
    "revision",
    "revision_range",
}


def _validate_commit_message_arguments(
    arguments: dict[str, Any],
    base_revision: str,
    head_revision: str,
) -> None:
    unexpected = set(arguments) - _COMMIT_MESSAGE_ARGUMENTS
    if unexpected:
        raise ToolGatewayError(
            "unsupported read_commit_messages argument(s): "
            + ", ".join(sorted(str(item) for item in unexpected))
        )
    for key in ("base_revision", "base"):
        if key in arguments and arguments[key] != base_revision:
            raise ToolGatewayError(
                f"unauthorized revision binding for {key}: {arguments[key]}"
            )
    for key in ("head_revision", "head"):
        if key in arguments and arguments[key] != head_revision:
            raise ToolGatewayError(
                f"unauthorized revision binding for {key}: {arguments[key]}"
            )
    expected_range = f"{base_revision}..{head_revision}"
    for key in ("revision", "revision_range"):
        if key not in arguments:
            continue
        if arguments[key] not in {expected_range, "base..head"}:
            raise ToolGatewayError(
                f"unauthorized revision binding for {key}: {arguments[key]}"
            )


def _parse_commit_messages(
    raw_log: str,
    *,
    max_body_chars: int,
) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    for record in raw_log.split("\x1e"):
        record = record.strip("\r\n")
        if not record:
            continue
        parts = record.split("\x00", 3)
        if len(parts) < 3:
            raise ToolGatewayError("git log returned an invalid commit record")
        commit_hash, subject, body = parts[:3]
        message_body, trailers = _split_commit_trailers(body)
        commits.append(
            {
                "hash": commit_hash.strip(),
                "subject": _bounded_commit_text(subject.strip(), 500),
                "body": _bounded_commit_text(message_body, max_body_chars),
                "trailers": [
                    {
                        "key": _bounded_commit_text(key, 200),
                        "value": _bounded_commit_text(value, 1000),
                    }
                    for key, value in trailers
                ],
            }
        )
    return commits


_TRAILER_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9-]*):[ \t]*(.+)$")


def _split_commit_trailers(body: str) -> tuple[str, list[tuple[str, str]]]:
    lines = body.rstrip().splitlines()
    trailers_reversed: list[tuple[str, str]] = []
    index = len(lines) - 1
    while index >= 0:
        match = _TRAILER_PATTERN.match(lines[index])
        if match is None:
            break
        trailers_reversed.append((match.group(1), match.group(2).strip()))
        index -= 1
    if not trailers_reversed:
        return body.strip(), []
    remaining = lines[: index + 1]
    while remaining and not remaining[-1].strip():
        remaining.pop()
    return "\n".join(remaining).strip(), list(reversed(trailers_reversed))


def _bounded_commit_text(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    marker = "\n[truncated]"
    return content[: max(0, max_chars - len(marker))] + marker


def _context_view(content: str, max_chars: int) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False
    return content[:max_chars] + "\n[truncated]\n", True
