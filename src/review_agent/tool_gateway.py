from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from collections.abc import Iterable
from typing import Any

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
    ) -> None:
        self.repository_path = repository_path
        self.base_revision = base_revision
        self.head_revision = head_revision
        self.observation_store = observation_store
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
        self.attempted_tool_calls = 0
        self.denied_tool_calls = 0

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
        self.denied_tool_calls += 1
        raise ToolGatewayError(
            f"unsupported tool: {tool_name}",
            code="unsupported_tool",
            tool_name=tool_name,
        )

    def _read_range(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        path = _safe_repo_path(str(arguments["path"]))
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
        path = _safe_repo_path(str(arguments["path"]))
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
        query = str(arguments["query"])
        if not query:
            raise ToolGatewayError("search query must not be empty")
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
        path = _safe_repo_path(str(arguments["path"]))
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
        name = str(arguments["name"])
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
        name = str(arguments["name"])
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


def _safe_repo_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not normalized or pure.parts[0] in {".git", ".env"}:
        raise ToolGatewayError(f"unsafe repository path: {path}")
    return normalized


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
