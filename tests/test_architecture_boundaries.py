from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "review_agent"

MEMORY_FOUNDATION_MODULES = (
    "memory_models.py",
    "memory_identity.py",
    "memory_store.py",
    "memory_sources.py",
    "repository_cache.py",
    "memory_retrieval.py",
    "memory_policy.py",
    "memory_feedback.py",
)

REVIEW_EXECUTION_MODULES = (
    "reviewer.py",
    "agent_loop.py",
    "orchestrator.py",
    "reviewer_task_executor.py",
    "tool_gateway.py",
)

V6_PROTOCOL_FORBIDDEN_MODULES = {
    "review_agent.pipeline",
    "review_agent.review_pipeline",
    "review_agent.memory_store",
    "review_agent.provider",
    "review_agent.model_adapter",
    "review_agent.model_adapter_factory",
    "review_agent_eval",
}


def _module_path(filename: str) -> Path:
    return SOURCE_ROOT / filename


def _tree(filename: str) -> ast.Module:
    module = _module_path(filename)
    return ast.parse(module.read_text(encoding="utf-8"), filename=str(module))


def _imports(tree: ast.AST) -> tuple[tuple[str, str | None], ...]:
    imported: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.extend((node.module, alias.name) for alias in node.names)
    return tuple(imported)


def _module_matches(module: str, forbidden: Iterable[str]) -> bool:
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in forbidden
    )


def _qualified_name(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return (*parent, node.attr) if parent else (node.attr,)
    return ()


def _calls(tree: ast.AST) -> tuple[tuple[tuple[str, ...], ast.Call], ...]:
    return tuple(
        (_qualified_name(node.func), node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )


def _identifier_tokens(identifier: str) -> frozenset[str]:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", identifier)
    return frozenset(
        token.casefold()
        for token in re.split(r"[^A-Za-z0-9]+", snake_case)
        if token
    )


def _definitions(tree: ast.AST) -> tuple[str, ...]:
    return tuple(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _class_definitions(tree: ast.AST) -> tuple[str, ...]:
    return tuple(
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    )


def _semantic_definitions(
    tree: ast.AST,
    *,
    action_tokens: set[str],
    domain_tokens: set[str],
) -> tuple[str, ...]:
    return tuple(
        name
        for name in _definitions(tree)
        if _identifier_tokens(name).intersection(action_tokens)
        and _identifier_tokens(name).intersection(domain_tokens)
    )


def _live_memory_store_bindings(tree: ast.AST) -> tuple[str, ...]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return tuple(
        sorted(
            name
            for name in names
            if {"memory", "store"} <= _identifier_tokens(name)
        )
    )


def _direct_store_transition_calls(tree: ast.AST) -> tuple[str, ...]:
    action_tokens = {
        "approve",
        "put",
        "reject",
        "revalidate",
        "revoke",
        "supersede",
        "transition",
        "write",
    }
    domain_tokens = {"candidate", "memory", "record"}
    violations: list[str] = []
    for qualified_name, _call in _calls(tree):
        if len(qualified_name) < 2:
            continue
        receiver_tokens = set().union(
            *(_identifier_tokens(part) for part in qualified_name[:-1])
        )
        method_tokens = _identifier_tokens(qualified_name[-1])
        if (
            "store" in receiver_tokens
            and method_tokens.intersection(action_tokens)
            and method_tokens.intersection(domain_tokens)
        ):
            violations.append(".".join(qualified_name))
    return tuple(sorted(set(violations)))


def _sql_literals(tree: ast.AST) -> tuple[str, ...]:
    statement_starters = {
        "ALTER",
        "CREATE",
        "DELETE",
        "DROP",
        "INSERT",
        "PRAGMA",
        "REPLACE",
        "SELECT",
        "UPDATE",
        "WITH",
    }
    statements: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        stripped = node.value.lstrip()
        if not stripped:
            continue
        first_token = stripped.split(None, 1)[0].rstrip("(").upper()
        if first_token in statement_starters:
            statements.append(stripped.splitlines()[0][:120])
    return tuple(statements)


def _database_execution_calls(tree: ast.AST) -> tuple[str, ...]:
    sql_methods = {"cursor", "execute", "executemany", "executescript"}
    database_tokens = {"connection", "cursor", "database", "db", "sqlite"}
    violations: list[str] = []
    for qualified_name, _call in _calls(tree):
        if len(qualified_name) < 2 or qualified_name[-1] not in sql_methods:
            continue
        receiver_tokens = set().union(
            *(_identifier_tokens(part) for part in qualified_name[:-1])
        )
        if receiver_tokens.intersection(database_tokens):
            violations.append(".".join(qualified_name))
    return tuple(sorted(set(violations)))


def test_reviewer_business_modules_do_not_import_legacy_model_provider() -> None:
    forbidden_modules = {"review_agent.provider"}
    forbidden_symbols = {"ModelProvider", "build_provider_from_config"}

    for filename in ("reviewer.py", "orchestrator.py", "cli.py", "agent_loop.py"):
        tree = _tree(filename)
        imported = _imports(tree)
        assert not [
            (module, symbol)
            for module, symbol in imported
            if _module_matches(module, forbidden_modules)
            or symbol in forbidden_symbols
        ], filename
        assert not set(_definitions(tree)).intersection(forbidden_symbols), filename


def test_v6_review_protocol_does_not_depend_on_runtime_or_eval_layers() -> None:
    violations = [
        (module, symbol)
        for module, symbol in _imports(_tree("review_protocol.py"))
        if _module_matches(module, V6_PROTOCOL_FORBIDDEN_MODULES)
        or symbol in {
            "MemoryStore",
            "ModelAdapter",
            "ModelAdapterFactory",
            "Pipeline",
            "ReviewPipelineV6",
        }
    ]

    assert not violations


def test_v6_reviewer_context_uses_frozen_memory_not_live_store_or_legacy_models() -> None:
    forbidden_modules = {
        "review_agent.memory_store",
        "review_agent.models",
        "review_agent.pipeline",
        "review_agent.provider",
        "review_agent_eval",
    }

    for filename in (
        "global_memory.py",
        "review_policy.py",
        "review_context.py",
    ):
        tree = _tree(filename)
        violations = [
            (module, symbol)
            for module, symbol in _imports(tree)
            if _module_matches(module, forbidden_modules)
            or symbol == "MemoryStore"
        ]
        assert not violations, f"{filename}: {violations}"
        assert not _live_memory_store_bindings(tree), filename


def test_quality_business_logic_does_not_own_subprocess_execution() -> None:
    quality = _tree("quality.py")
    pipeline = _tree("pipeline.py")

    for filename, tree in (("quality.py", quality), ("pipeline.py", pipeline)):
        assert not [
            imported
            for imported in _imports(tree)
            if imported == ("subprocess", "Popen")
        ], filename
        assert ("subprocess", "Popen") not in {
            qualified_name for qualified_name, _call in _calls(tree)
        }, filename
        assert "_run_bounded_process" not in _definitions(tree), filename


def test_memory_foundations_do_not_depend_on_pipeline_cli_or_model_adapters() -> None:
    forbidden_modules = {
        "review_agent.pipeline",
        "review_agent.command",
        "review_agent.cli",
        "review_agent.provider",
        "review_agent.model_adapter",
        "review_agent.model_adapter_factory",
    }

    for filename in MEMORY_FOUNDATION_MODULES:
        violations = [
            (module, symbol)
            for module, symbol in _imports(_tree(filename))
            if _module_matches(module, forbidden_modules)
        ]
        assert not violations, f"{filename}: forbidden imports {violations}"


def test_memory_store_depends_only_on_memory_foundations_and_standard_library() -> None:
    review_agent_imports = {
        module
        for module, _symbol in _imports(_tree("memory_store.py"))
        if module.startswith("review_agent.")
    }

    assert review_agent_imports <= {
        "review_agent.memory_identity",
        "review_agent.memory_models",
    }


def test_memory_curator_imports_only_the_adapter_factory_protocol() -> None:
    imported = _imports(_tree("memory_curator.py"))
    forbidden_modules = {
        "review_agent.provider",
        "review_agent.model_adapter",
    }
    violations = [
        (module, symbol)
        for module, symbol in imported
        if _module_matches(module, forbidden_modules)
        or (
            module == "review_agent.model_adapter_factory"
            and symbol != "ModelAdapterFactory"
        )
    ]

    assert not violations
    assert (
        "review_agent.model_adapter_factory",
        "ModelAdapterFactory",
    ) in imported


def test_pipeline_delegates_memory_sql_scanning_lifecycle_and_ranking() -> None:
    tree = _tree("pipeline.py")
    imported_modules = {module for module, _symbol in _imports(tree)}

    assert not any(
        _module_matches(module, {"sqlite3"}) for module in imported_modules
    )
    assert not _sql_literals(tree)
    assert not _database_execution_calls(tree)
    assert not _direct_store_transition_calls(tree)

    assert not _semantic_definitions(
        tree,
        action_tokens={"classify", "detect", "mask", "redact", "sanitize", "scan", "scrub"},
        domain_tokens={"credential", "injection", "secret", "sensitive", "token"},
    )
    assert not _semantic_definitions(
        tree,
        action_tokens={"approve", "reject", "revalidate", "revoke", "supersede", "transition"},
        domain_tokens={"approval", "candidate", "lifecycle", "memory", "record"},
    )
    assert not _semantic_definitions(
        tree,
        action_tokens={"prioritize", "rank", "relevance", "score", "weight"},
        domain_tokens={"candidate", "context", "memory", "record", "retrieval"},
    )


def test_cli_delegates_memory_lifecycle_policy_and_retrieval() -> None:
    tree = _tree("command.py")
    forbidden_implementation_modules = {
        "review_agent.memory_policy",
        "review_agent.memory_retrieval",
    }
    forbidden_imports = [
        (module, symbol)
        for module, symbol in _imports(tree)
        if _module_matches(module, forbidden_implementation_modules)
    ]

    assert not forbidden_imports
    assert not _direct_store_transition_calls(tree)
    assert not _semantic_definitions(
        tree,
        action_tokens={"compile", "evaluate", "prioritize", "rank", "retrieve", "score"},
        domain_tokens={"applicability", "context", "memory", "policy", "record", "retrieval"},
    )
    assert not [
        name
        for name in _class_definitions(tree)
        if _identifier_tokens(name).intersection(
            {"compiler", "lifecycle", "ranker", "retriever"}
        )
    ]


def test_reviewer_execution_modules_cannot_hold_a_live_memory_store() -> None:
    for filename in REVIEW_EXECUTION_MODULES:
        tree = _tree(filename)
        imports = _imports(tree)
        forbidden_imports = [
            (module, symbol)
            for module, symbol in imports
            if _module_matches(module, {"review_agent.memory_store"})
            or symbol == "MemoryStore"
        ]
        assert not forbidden_imports, filename
        assert not _live_memory_store_bindings(tree), filename
