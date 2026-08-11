from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess
import sys
from typing import Any, Mapping

from review_agent.memory_identity import repository_key as canonical_repository_key
from review_agent.memory_models import (
    RepositoryKnowledgeCapability,
    canonical_json,
    canonical_sha256,
)
from review_agent.repository_cache import (
    CAPABILITY_METADATA,
    RepositoryCacheProvenance,
    RepositoryKnowledgeArtifact,
    RepositoryKnowledgeCache,
    build_repository_knowledge_key,
)
from review_agent.revision import RevisionResolver, sanitized_git_environment


REPOSITORY_INTELLIGENCE_ARTIFACT_SCHEMA = "repository_intelligence_symbols_v1"
REPOSITORY_INTELLIGENCE_ANALYZER_NAME = "python-ast"
REPOSITORY_INTELLIGENCE_ANALYZER_VERSION = "repository-intelligence-v1"
CHANGED_SYMBOLS_V2_ANALYZER_VERSION = "python-ast-v2"
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:/")


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
class LanguageCoverageV2:
    language: str
    status: str
    reason_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "status": self.status,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class ChangedSymbolV2:
    path: str
    qualified_name: str
    kind: str
    change_type: str
    line_start: int
    line_end: int
    analyzer: str
    analyzer_version: str
    analysis_configuration: str
    language_coverage: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ChangedSymbolsV2:
    snapshot_id: str
    base_sha: str
    head_sha: str
    analyzer: str
    analyzer_version: str
    analysis_configuration: str
    cache_key: str
    language_coverage: tuple[LanguageCoverageV2, ...]
    symbols: tuple[ChangedSymbolV2, ...]

    @classmethod
    def empty(
        cls,
        *,
        snapshot_id: str,
        base_sha: str,
        head_sha: str,
        changed_files: list[str],
        coverage_status: str = "not_analyzed",
        reason_code: str = "analyzer_not_run",
    ) -> "ChangedSymbolsV2":
        configuration = canonical_json({})
        languages = sorted({_language_for_path(path) for path in changed_files})
        coverage = tuple(
            LanguageCoverageV2(
                language=language,
                status=coverage_status,
                reason_code=reason_code,
            )
            for language in languages
        )
        cache_key = hashlib.sha256(
            canonical_json(
                {
                    "snapshot_id": snapshot_id,
                    "base_sha": base_sha,
                    "head_sha": head_sha,
                    "analyzer": REPOSITORY_INTELLIGENCE_ANALYZER_NAME,
                    "analyzer_version": CHANGED_SYMBOLS_V2_ANALYZER_VERSION,
                    "analysis_configuration": configuration,
                }
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            snapshot_id=snapshot_id,
            base_sha=base_sha,
            head_sha=head_sha,
            analyzer=REPOSITORY_INTELLIGENCE_ANALYZER_NAME,
            analyzer_version=CHANGED_SYMBOLS_V2_ANALYZER_VERSION,
            analysis_configuration=configuration,
            cache_key=cache_key,
            language_coverage=coverage,
            symbols=(),
        )


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

    @property
    def cache_provenance(self) -> RepositoryCacheProvenance | None:
        # Provenance is Session-phase metadata, not part of the long-standing
        # authoritative snapshot schema.  Keeping it outside dataclass fields
        # also preserves compatibility for callers that use dataclasses.asdict.
        return getattr(self, "_cache_provenance", None)


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
    *,
    cache_backend: RepositoryKnowledgeCache | None = None,
    repository_key: str | None = None,
    review_id: str | None = None,
    lsp_status: str = "unavailable",
    fallback_strategy: str = "python_ast+git_grep",
    text_search_backend: str = "git-grep",
    analyzer_name: str = REPOSITORY_INTELLIGENCE_ANALYZER_NAME,
    analyzer_version: str = REPOSITORY_INTELLIGENCE_ANALYZER_VERSION,
    python_ast_version: str | None = None,
    text_search_backend_version: str | None = None,
    analysis_configuration: Mapping[str, object] | None = None,
) -> RepositoryIntelligenceSnapshot:
    normalized_changed_files = _canonical_changed_files(changed_files)
    cache_provenance = None
    authoritative_base = base_revision
    authoritative_head = head_revision
    if cache_backend is None:
        changed_symbols = detect_changed_symbols(
            repo,
            base_revision,
            head_revision,
            normalized_changed_files,
        )
    else:
        if analysis_configuration is not None and not isinstance(
            analysis_configuration,
            Mapping,
        ):
            raise ValueError("analysis_configuration must be a mapping")
        resolver = RevisionResolver()
        resolved_revisions = resolver.resolve_pair(
            repo,
            base_revision,
            head_revision,
        )
        authoritative_base = resolved_revisions.resolved_base_sha.casefold()
        authoritative_head = resolved_revisions.resolved_head_sha.casefold()
        actual_repository_key = canonical_repository_key(
            resolver.repository_identity(repo)
        )
        if repository_key is not None and repository_key != actual_repository_key:
            raise ValueError(
                "repository_key does not match the authorized repository identity"
            )
        resolved_repository_key = actual_repository_key
        resolved_ast_version = python_ast_version or _python_ast_version()
        resolved_text_version = text_search_backend_version or _text_backend_version(
            repo,
            text_search_backend,
        )
        configuration = {
            "schema": "repository_intelligence_configuration_v1",
            "lsp_status": _required_configuration_text(lsp_status, "lsp_status"),
            "fallback_strategy": _required_configuration_text(
                fallback_strategy,
                "fallback_strategy",
            ),
            "text_search_backend": _required_configuration_text(
                text_search_backend,
                "text_search_backend",
            ),
            "python_ast_version": _required_configuration_text(
                resolved_ast_version,
                "python_ast_version",
            ),
            "text_search_backend_version": _required_configuration_text(
                resolved_text_version,
                "text_search_backend_version",
            ),
            "analysis_configuration": dict(analysis_configuration or {}),
        }
        inputs = {
            "schema": "repository_intelligence_input_v1",
            "changed_files": normalized_changed_files,
        }
        key = build_repository_knowledge_key(
            repository_key=resolved_repository_key,
            revision_binding=f"{authoritative_base}..{authoritative_head}",
            capability=RepositoryKnowledgeCapability.SYMBOL_INDEX,
            analyzer_name=analyzer_name,
            analyzer_version=analyzer_version,
            configuration=configuration,
            inputs=inputs,
        )

        def build_symbols() -> RepositoryKnowledgeArtifact:
            symbols = detect_changed_symbols(
                repo,
                authoritative_base,
                authoritative_head,
                normalized_changed_files,
            )
            payload = _changed_symbols_payload(symbols)
            content = canonical_json(payload).encode("utf-8")
            metadata = CAPABILITY_METADATA[RepositoryKnowledgeCapability.SYMBOL_INDEX]
            return RepositoryKnowledgeArtifact(
                content=content,
                content_type=metadata.content_type,
                artifact_schema=REPOSITORY_INTELLIGENCE_ARTIFACT_SCHEMA,
                summary_hash=canonical_sha256(
                    {
                        "schema": "repository_intelligence_summary_v1",
                        "changed_symbol_count": len(symbols),
                    }
                ),
            )

        cache_result = cache_backend.get_or_build(
            key,
            build_symbols,
            review_id=review_id,
            validator=_changed_symbols_from_content,
            fallback_provenance={
                "lsp_status": lsp_status,
                "fallback_strategy": fallback_strategy,
                "text_search_backend": text_search_backend,
            },
            content_type=CAPABILITY_METADATA[
                RepositoryKnowledgeCapability.SYMBOL_INDEX
            ].content_type,
            artifact_schema=REPOSITORY_INTELLIGENCE_ARTIFACT_SCHEMA,
        )
        if cache_result.content is None:  # pragma: no cover - API invariant
            raise RuntimeError("repository cache returned no authoritative content")
        changed_symbols = _changed_symbols_from_content(cache_result.content)
        cache_provenance = cache_result.provenance
    snapshot = RepositoryIntelligenceSnapshot(
        base_revision=authoritative_base,
        revision=authoritative_head,
        changed_symbols=changed_symbols,
        lsp_status=lsp_status,
        fallback_strategy=fallback_strategy,
        text_search_backend=text_search_backend,
    )
    if cache_provenance is not None:
        object.__setattr__(snapshot, "_cache_provenance", cache_provenance)
    return snapshot


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


def changed_symbols_cache_key(
    *,
    repository_key: str,
    base_sha: str,
    head_sha: str,
    analyzer: str,
    analyzer_version: str,
    analysis_configuration: Mapping[str, object],
) -> str:
    if not isinstance(repository_key, str) or re.fullmatch(
        r"[0-9a-f]{64}", repository_key
    ) is None:
        raise ValueError("ChangedSymbols repository_key is invalid")
    object_id = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
    if object_id.fullmatch(base_sha) is None or object_id.fullmatch(head_sha) is None:
        raise ValueError("ChangedSymbols revisions must be full Git object IDs")
    if not isinstance(analyzer, str) or not analyzer.strip():
        raise ValueError("ChangedSymbols analyzer is invalid")
    if not isinstance(analyzer_version, str) or not analyzer_version.strip():
        raise ValueError("ChangedSymbols analyzer version is invalid")
    if not isinstance(analysis_configuration, Mapping):
        raise ValueError("ChangedSymbols analysis configuration must be an object")
    return hashlib.sha256(
        canonical_json(
            {
                "schema": "changed_symbols_cache_key_v2",
                "repository_key": repository_key,
                "base_sha": base_sha,
                "head_sha": head_sha,
                "analyzer": analyzer,
                "analyzer_version": analyzer_version,
                "analysis_configuration": dict(analysis_configuration),
            }
        ).encode("utf-8")
    ).hexdigest()


def build_changed_symbols_v2(
    repo: Path,
    *,
    snapshot_id: str,
    base_sha: str,
    head_sha: str,
    changed_files: list[str],
    analyzer_version: str = CHANGED_SYMBOLS_V2_ANALYZER_VERSION,
    analysis_configuration: Mapping[str, object] | None = None,
) -> ChangedSymbolsV2:
    if not isinstance(snapshot_id, str) or re.fullmatch(
        r"S-[0-9a-f]{64}", snapshot_id
    ) is None:
        raise ValueError("ChangedSymbols snapshot_id is invalid")
    normalized_files = _canonical_changed_files(changed_files)
    configuration = dict(analysis_configuration or {})
    configuration_json = canonical_json(configuration)
    resolver = RevisionResolver()
    resolved = resolver.resolve_pair(repo, base_sha, head_sha)
    authoritative_base = resolved.resolved_base_sha.casefold()
    authoritative_head = resolved.resolved_head_sha.casefold()
    repository_key = canonical_repository_key(resolver.repository_identity(repo))
    cache_key = changed_symbols_cache_key(
        repository_key=repository_key,
        base_sha=authoritative_base,
        head_sha=authoritative_head,
        analyzer=REPOSITORY_INTELLIGENCE_ANALYZER_NAME,
        analyzer_version=analyzer_version,
        analysis_configuration=configuration,
    )

    python_files = [path for path in normalized_files if _language_for_path(path) == "python"]
    covered_python_files: list[str] = []
    failed_python_files: list[str] = []
    for path in python_files:
        parse_failed = False
        for revision in (authoritative_base, authoritative_head):
            content = _git_show(repo, revision, path, allow_missing=True)
            if content is None:
                continue
            try:
                ast.parse(content)
            except SyntaxError:
                parse_failed = True
                break
        if parse_failed:
            failed_python_files.append(path)
        else:
            covered_python_files.append(path)

    detected = detect_changed_symbols(
        repo,
        authoritative_base,
        authoritative_head,
        covered_python_files,
    )
    symbols = tuple(
        ChangedSymbolV2(
            path=symbol.path,
            qualified_name=symbol.qualified_name,
            kind=symbol.kind,
            change_type=symbol.change_type,
            line_start=symbol.line_start,
            line_end=symbol.line_end,
            analyzer=REPOSITORY_INTELLIGENCE_ANALYZER_NAME,
            analyzer_version=analyzer_version,
            analysis_configuration=configuration_json,
            language_coverage="covered",
        )
        for symbol in detected
    )

    coverage: list[LanguageCoverageV2] = []
    languages = sorted({_language_for_path(path) for path in normalized_files})
    for language in languages:
        if language == "python":
            if failed_python_files:
                coverage.append(
                    LanguageCoverageV2(
                        language="python",
                        status="partial",
                        reason_code="python_parse_failed",
                    )
                )
            else:
                coverage.append(
                    LanguageCoverageV2(
                        language="python",
                        status="covered",
                    )
                )
        else:
            coverage.append(
                LanguageCoverageV2(
                    language=language,
                    status="not_analyzed",
                    reason_code="analyzer_language_unsupported",
                )
            )
    return ChangedSymbolsV2(
        snapshot_id=snapshot_id,
        base_sha=authoritative_base,
        head_sha=authoritative_head,
        analyzer=REPOSITORY_INTELLIGENCE_ANALYZER_NAME,
        analyzer_version=analyzer_version,
        analysis_configuration=configuration_json,
        cache_key=cache_key,
        language_coverage=tuple(coverage),
        symbols=symbols,
    )


def changed_symbols_v2_to_dict(result: ChangedSymbolsV2) -> dict[str, object]:
    if not isinstance(result, ChangedSymbolsV2):
        raise ValueError("ChangedSymbols v2 result is invalid")
    return {
        "schema_version": "changed_symbols_v2",
        "snapshot_id": result.snapshot_id,
        "base_sha": result.base_sha,
        "head_sha": result.head_sha,
        "analyzer": result.analyzer,
        "analyzer_version": result.analyzer_version,
        "analysis_configuration": result.analysis_configuration,
        "cache_key": result.cache_key,
        "language_coverage": [
            coverage.to_dict() for coverage in result.language_coverage
        ],
        "symbols": [symbol.to_dict() for symbol in result.symbols],
    }


def changed_symbols_v2_from_dict(payload: Mapping[str, Any]) -> ChangedSymbolsV2:
    expected = {
        "schema_version",
        "snapshot_id",
        "base_sha",
        "head_sha",
        "analyzer",
        "analyzer_version",
        "analysis_configuration",
        "cache_key",
        "language_coverage",
        "symbols",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("ChangedSymbols v2 artifact schema is invalid")
    if payload["schema_version"] != "changed_symbols_v2":
        raise ValueError("ChangedSymbols v2 artifact version is unsupported")
    if type(payload["language_coverage"]) is not list or type(
        payload["symbols"]
    ) is not list:
        raise ValueError("ChangedSymbols v2 arrays are invalid")
    try:
        coverage = tuple(
            LanguageCoverageV2(**_exact_changed_symbol_row(
                item,
                {"language", "status", "reason_code"},
                "language coverage",
            ))
            for item in payload["language_coverage"]
        )
        symbols = tuple(
            ChangedSymbolV2(**_exact_changed_symbol_row(
                item,
                {
                    "path",
                    "qualified_name",
                    "kind",
                    "change_type",
                    "line_start",
                    "line_end",
                    "analyzer",
                    "analyzer_version",
                    "analysis_configuration",
                    "language_coverage",
                },
                "changed symbol",
            ))
            for item in payload["symbols"]
        )
        result = ChangedSymbolsV2(
            snapshot_id=payload["snapshot_id"],
            base_sha=payload["base_sha"],
            head_sha=payload["head_sha"],
            analyzer=payload["analyzer"],
            analyzer_version=payload["analyzer_version"],
            analysis_configuration=payload["analysis_configuration"],
            cache_key=payload["cache_key"],
            language_coverage=coverage,
            symbols=symbols,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("ChangedSymbols v2 artifact is invalid") from error
    if changed_symbols_v2_to_dict(result) != dict(payload):
        raise ValueError("ChangedSymbols v2 artifact is not canonical")
    if re.fullmatch(r"S-[0-9a-f]{64}", result.snapshot_id) is None:
        raise ValueError("ChangedSymbols v2 Snapshot ID is invalid")
    object_id = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
    if (
        object_id.fullmatch(result.base_sha) is None
        or object_id.fullmatch(result.head_sha) is None
        or re.fullmatch(r"[0-9a-f]{64}", result.cache_key) is None
    ):
        raise ValueError("ChangedSymbols v2 binding is invalid")
    return result


def _exact_changed_symbol_row(
    value: Any,
    expected: set[str],
    context: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"ChangedSymbols v2 {context} schema is invalid")
    return dict(value)


def _language_for_path(path: str) -> str:
    suffix = Path(path).suffix.casefold()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
    }.get(suffix, "other")


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
    # Cache provenance is recorded separately by the owning Session phase.  The
    # authoritative Repository Intelligence artifact retains its established
    # strict shape so v1-v4 hydration and resume remain byte-compatible.
    return {
        "base_revision": snapshot.base_revision,
        "revision": snapshot.revision,
        "changed_symbols": [asdict(symbol) for symbol in snapshot.changed_symbols],
        "lsp_status": snapshot.lsp_status,
        "fallback_strategy": snapshot.fallback_strategy,
        "text_search_backend": snapshot.text_search_backend,
    }


def repository_intelligence_raw_json(snapshot: RepositoryIntelligenceSnapshot) -> str:
    return json.dumps(repository_intelligence_to_dict(snapshot), ensure_ascii=False, indent=2)


def summarize_repository_intelligence(snapshot: RepositoryIntelligenceSnapshot) -> str:
    if snapshot.lsp_status == "unavailable":
        analyzer_line = f"LSP unavailable; using {snapshot.fallback_strategy}"
    else:
        analyzer_line = f"LSP {snapshot.lsp_status}; using {snapshot.fallback_strategy}"
    lines = [
        "Repository Intelligence",
        f"Revision: {snapshot.revision}",
        analyzer_line,
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


def _canonical_changed_files(changed_files: list[str]) -> list[str]:
    if not isinstance(changed_files, list):
        raise ValueError("changed_files must be a list")
    normalized: set[str] = set()
    for path in changed_files:
        if not isinstance(path, str) or not path.strip() or path != path.strip():
            raise ValueError("changed_files must contain non-empty repository paths")
        canonical = path.replace("\\", "/")
        parts = canonical.split("/")
        if (
            canonical.startswith("/")
            or _WINDOWS_DRIVE_PATH.match(canonical)
            or ".." in parts
            or "" in parts
            or parts[0].casefold() == ".git"
        ):
            raise ValueError("changed_files must contain safe repository-relative paths")
        normalized.add(canonical)
    return sorted(normalized)


def _changed_symbols_payload(symbols: list[ChangedSymbol]) -> dict[str, object]:
    return {
        "schema": REPOSITORY_INTELLIGENCE_ARTIFACT_SCHEMA,
        "changed_symbols": [asdict(symbol) for symbol in symbols],
    }


def _changed_symbols_from_content(content: bytes) -> list[ChangedSymbol]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("repository intelligence cache payload is not valid JSON") from None
    if not isinstance(payload, dict) or set(payload) != {"schema", "changed_symbols"}:
        raise ValueError("repository intelligence cache payload has an invalid envelope")
    if payload["schema"] != REPOSITORY_INTELLIGENCE_ARTIFACT_SCHEMA:
        raise ValueError("repository intelligence cache schema is unsupported")
    rows = payload["changed_symbols"]
    if not isinstance(rows, list):
        raise ValueError("repository intelligence changed_symbols must be a list")

    symbols: list[ChangedSymbol] = []
    expected_fields = {
        "path",
        "qualified_name",
        "kind",
        "change_type",
        "line_start",
        "line_end",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("repository intelligence cached symbol is invalid")
        text_fields = ("path", "qualified_name", "kind", "change_type")
        if any(
            not isinstance(row[name], str)
            or not row[name].strip()
            or row[name] != row[name].strip()
            for name in text_fields
        ):
            raise ValueError("repository intelligence cached symbol text is invalid")
        if row["change_type"] not in {"added", "modified", "deleted"}:
            raise ValueError("repository intelligence cached change type is invalid")
        if (
            type(row["line_start"]) is not int
            or type(row["line_end"]) is not int
            or row["line_start"] < 1
            or row["line_end"] < row["line_start"]
        ):
            raise ValueError("repository intelligence cached line range is invalid")
        path = row["path"].replace("\\", "/")
        path_parts = path.split("/")
        if (
            path != row["path"]
            or path.startswith("/")
            or _WINDOWS_DRIVE_PATH.match(path)
            or ".." in path_parts
            or "" in path_parts
            or path_parts[0].casefold() == ".git"
        ):
            raise ValueError("repository intelligence cached path is invalid")
        symbols.append(
            ChangedSymbol(
                path=row["path"],
                qualified_name=row["qualified_name"],
                kind=row["kind"],
                change_type=row["change_type"],
                line_start=row["line_start"],
                line_end=row["line_end"],
            )
        )

    canonical_order = sorted(
        symbols,
        key=lambda item: (
            item.path,
            item.line_start,
            item.qualified_name,
            item.change_type,
        ),
    )
    if symbols != canonical_order or len(set(map(_changed_symbol_identity, symbols))) != len(
        symbols
    ):
        raise ValueError("repository intelligence cached symbols are not canonical")
    if canonical_json(payload).encode("utf-8") != content:
        raise ValueError("repository intelligence cache JSON is not canonical")
    return symbols


def _changed_symbol_identity(symbol: ChangedSymbol) -> tuple[object, ...]:
    return (
        symbol.path,
        symbol.qualified_name,
        symbol.kind,
        symbol.change_type,
        symbol.line_start,
        symbol.line_end,
    )


def _python_ast_version() -> str:
    return "cpython-%d.%d.%d-ast-v1" % (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )


def _text_backend_version(repo: Path, backend: str) -> str:
    if backend == "git-grep":
        return _run_git(repo, ["--version"], allow_exit_codes={0}).strip()
    if backend in {"rg", "ripgrep"}:
        try:
            result = subprocess.run(
                ["rg", "--version"],
                cwd=repo,
                env=sanitized_git_environment(),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "ripgrep-version-unavailable"
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.splitlines()[0].strip()
        return "ripgrep-version-unavailable"
    return backend + "-version-unspecified"


def _required_configuration_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _list_python_files(repo: Path, revision: str) -> list[str]:
    raw = _run_git(repo, ["ls-tree", "-r", "--name-only", revision], allow_exit_codes={0})
    return [line for line in raw.splitlines() if line.endswith(".py")]


def _git_show(repo: Path, revision: str, path: str, allow_missing: bool) -> str | None:
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "show", f"{revision}:{path}"],
            cwd=repo,
            env=sanitized_git_environment(),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError(f"git show failed for {path}") from None
    if result.returncode == 0:
        return result.stdout.lstrip("\ufeff")
    if allow_missing:
        return None
    raise RuntimeError(result.stderr.strip() or f"git show failed for {path}")


def _run_git(repo: Path, args: list[str], allow_exit_codes: set[int]) -> str:
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", *args],
            cwd=repo,
            env=sanitized_git_environment(),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError(f"git {' '.join(args)} failed") from None
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
