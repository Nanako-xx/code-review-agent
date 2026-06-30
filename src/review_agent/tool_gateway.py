from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any

from review_agent.observations import ObservationStore
from review_agent.repository_intelligence import collect_python_symbols, search_repository_text


class ToolGatewayError(ValueError):
    pass


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    observation_ids: list[str]
    context_view: str
    truncated: bool = False


class ToolGateway:
    def __init__(
        self,
        repository_path: Path,
        base_revision: str,
        head_revision: str,
        observation_store: ObservationStore,
        max_context_chars: int = 4000,
        timeout_seconds: int = 10,
    ) -> None:
        self.repository_path = repository_path
        self.base_revision = base_revision
        self.head_revision = head_revision
        self.observation_store = observation_store
        self.max_context_chars = max_context_chars
        self.timeout_seconds = timeout_seconds

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
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
        raise ToolGatewayError(f"unsupported tool: {tool_name}")

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


def _context_view(content: str, max_chars: int) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False
    return content[:max_chars] + "\n[truncated]\n", True
