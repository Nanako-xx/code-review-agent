from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import ast
import hashlib
import json
import subprocess


@dataclass(frozen=True)
class PythonSymbol:
    path: str
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int
    body_hash: str
    calls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChangedSymbol:
    path: str
    qualified_name: str
    kind: str
    change_type: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class TextSearchMatch:
    path: str
    line_number: int
    line: str


@dataclass(frozen=True)
class RepositoryIntelligenceSnapshot:
    base_revision: str
    revision: str
    changed_symbols: list[ChangedSymbol]
    lsp_status: str = "unavailable"
    fallback_strategy: str = "python_ast+git_grep"
    text_search_backend: str = "git-grep"


def collect_python_symbols(repo: Path, revision: str, paths: list[str] | None = None) -> list[PythonSymbol]:
    candidate_paths = paths if paths is not None else _list_python_files(repo, revision)
    symbols: list[PythonSymbol] = []
    for path in candidate_paths:
        if not path.endswith(".py"):
            continue
        content = _git_show(repo, revision, path, allow_missing=True)
        if content is None:
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        symbols.extend(_symbols_from_tree(path, content, tree))
    return symbols


def build_repository_intelligence(
    repo: Path,
    base_revision: str,
    head_revision: str,
    changed_files: list[str],
) -> RepositoryIntelligenceSnapshot:
    changed_symbols = detect_changed_symbols(repo, base_revision, head_revision, changed_files)
    return RepositoryIntelligenceSnapshot(
        base_revision=base_revision,
        revision=head_revision,
        changed_symbols=changed_symbols,
    )


def detect_changed_symbols(
    repo: Path,
    base_revision: str,
    head_revision: str,
    changed_files: list[str],
) -> list[ChangedSymbol]:
    python_files = [path for path in changed_files if path.endswith(".py")]
    base_symbols = {
        (symbol.path, symbol.qualified_name): symbol
        for symbol in collect_python_symbols(repo, base_revision, python_files)
    }
    head_symbols = {
        (symbol.path, symbol.qualified_name): symbol
        for symbol in collect_python_symbols(repo, head_revision, python_files)
    }
    changes: list[ChangedSymbol] = []
    for key, symbol in head_symbols.items():
        previous = base_symbols.get(key)
        if previous is None:
            changes.append(_changed(symbol, "added"))
        elif previous.body_hash != symbol.body_hash:
            changes.append(_changed(symbol, "modified"))
    for key, symbol in base_symbols.items():
        if key not in head_symbols:
            changes.append(_changed(symbol, "deleted"))
    return sorted(changes, key=lambda item: (item.path, item.line_start, item.qualified_name, item.change_type))


def search_repository_text(repo: Path, revision: str, query: str, max_results: int = 20) -> list[TextSearchMatch]:
    if not query:
        return []
    raw = _run_git(repo, ["grep", "-n", "--fixed-strings", query, revision, "--", "."], allow_exit_codes={0, 1})
    matches: list[TextSearchMatch] = []
    for line in raw.splitlines()[:max_results]:
        parsed = _parse_git_grep_line(line)
        if parsed is not None:
            matches.append(parsed)
    return matches


def repository_intelligence_to_dict(snapshot: RepositoryIntelligenceSnapshot) -> dict[str, object]:
    return asdict(snapshot)


def repository_intelligence_raw_json(snapshot: RepositoryIntelligenceSnapshot) -> str:
    return json.dumps(repository_intelligence_to_dict(snapshot), ensure_ascii=False, indent=2)


def summarize_repository_intelligence(snapshot: RepositoryIntelligenceSnapshot) -> str:
    lines = [
        "Repository Intelligence",
        f"Revision: {snapshot.revision}",
        f"LSP unavailable; using {snapshot.fallback_strategy}",
        f"Text search backend: {snapshot.text_search_backend}",
        "Changed Symbols:",
    ]
    if not snapshot.changed_symbols:
        lines.append("- No changed Python symbols detected")
    for symbol in snapshot.changed_symbols:
        lines.append(
            f"- {symbol.change_type} {symbol.kind} {symbol.qualified_name} "
            f"{symbol.path}:{symbol.line_start}-{symbol.line_end}"
        )
    return "\n".join(lines)


def _symbols_from_tree(path: str, content: str, tree: ast.AST) -> list[PythonSymbol]:
    lines = content.splitlines()
    collector = _SymbolCollector(path, lines)
    collector.visit(tree)
    return collector.symbols


class _SymbolCollector(ast.NodeVisitor):
    def __init__(self, path: str, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.stack: list[tuple[str, str]] = []
        self.symbols: list[PythonSymbol] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_symbol(node, "class")
        self.stack.append((node.name, "class"))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_symbol(node, "method" if self._inside_class() else "function")
        self.stack.append((node.name, "function"))
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_symbol(node, "method" if self._inside_class() else "async_function")
        self.stack.append((node.name, "function"))
        self.generic_visit(node)
        self.stack.pop()

    def _add_symbol(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
        line_start = int(node.lineno)
        line_end = int(getattr(node, "end_lineno", node.lineno))
        qualified_name = ".".join([name for name, _kind in self.stack] + [node.name])
        body = "\n".join(self.lines[line_start - 1 : line_end])
        self.symbols.append(
            PythonSymbol(
                path=self.path,
                name=node.name,
                qualified_name=qualified_name,
                kind=kind,
                line_start=line_start,
                line_end=line_end,
                body_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                calls=_call_names(node),
            )
        )

    def _inside_class(self) -> bool:
        return any(kind == "class" for _name, kind in self.stack)


def _call_names(node: ast.AST) -> list[str]:
    calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _call_name(child.func)
            if name:
                calls.add(name)
    return sorted(calls)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _changed(symbol: PythonSymbol, change_type: str) -> ChangedSymbol:
    return ChangedSymbol(
        path=symbol.path,
        qualified_name=symbol.qualified_name,
        kind=symbol.kind,
        change_type=change_type,
        line_start=symbol.line_start,
        line_end=symbol.line_end,
    )


def _list_python_files(repo: Path, revision: str) -> list[str]:
    raw = _run_git(repo, ["ls-tree", "-r", "--name-only", revision], allow_exit_codes={0})
    return [line for line in raw.splitlines() if line.endswith(".py")]


def _git_show(repo: Path, revision: str, path: str, allow_missing: bool) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        return result.stdout
    if allow_missing:
        return None
    raise RuntimeError(result.stderr.strip() or f"git show failed for {path}")


def _run_git(repo: Path, args: list[str], allow_exit_codes: set[int]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode not in allow_exit_codes:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def _parse_git_grep_line(line: str) -> TextSearchMatch | None:
    parts = line.split(":", 2)
    if len(parts) != 3:
        return None
    path, line_number, content = parts
    if line_number.isdigit():
        return TextSearchMatch(path=path, line_number=int(line_number), line=content)
    nested = content.split(":", 1)
    if len(nested) == 2 and nested[0].isdigit():
        return TextSearchMatch(path=line_number, line_number=int(nested[0]), line=nested[1])
    return None
