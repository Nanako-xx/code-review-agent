from pathlib import Path

import pytest

from conftest import run_git
from review_agent.hydration import repository_intelligence_from_dict
from review_agent.memory_identity import repository_key as canonical_repository_key
from review_agent.memory_models import MemoryMode
from review_agent.memory_store import MemoryStore
from review_agent.repository_cache import RepositoryCacheStatus, RepositoryKnowledgeCache
from review_agent.repository_intelligence import (
    build_repository_intelligence,
    collect_python_symbols,
    repository_intelligence_to_dict,
    search_repository_text,
    summarize_repository_intelligence,
)
from review_agent.revision import RevisionResolver


def _repository_key(repo: Path) -> str:
    return canonical_repository_key(RevisionResolver().repository_identity(repo))


def test_collect_python_symbols_reads_committed_head(git_repo: Path):
    (git_repo / "auth.py").write_text(
        "import os\n\n"
        "class User:\n"
        "    def is_admin(self):\n"
        "        return check_role(self.role)\n\n"
        "def check_role(role):\n"
        "    return role == 'admin'\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth symbols")
    head = run_git(git_repo, "rev-parse", "HEAD")

    symbols = collect_python_symbols(git_repo, head, paths=["auth.py"])
    by_name = {symbol.qualified_name: symbol for symbol in symbols}

    assert by_name["User"].kind == "class"
    assert by_name["User.is_admin"].kind == "method"
    assert by_name["User.is_admin"].calls == ["check_role"]
    assert by_name["check_role"].line_start == 7
    assert by_name["check_role"].line_end == 8


def test_build_repository_intelligence_detects_changed_symbols(git_repo: Path):
    (git_repo / "auth.py").write_text(
        "def check_role(role):\n"
        "    return role == 'admin'\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "base auth")
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text(
        "def check_role(role):\n"
        "    return role in {'admin', 'owner'}\n\n"
        "def is_owner(role):\n"
        "    return role == 'owner'\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "head auth")
    head = run_git(git_repo, "rev-parse", "HEAD")

    snapshot = build_repository_intelligence(git_repo, base, head, changed_files=["auth.py"])

    changed = {(item.qualified_name, item.change_type) for item in snapshot.changed_symbols}
    assert ("check_role", "modified") in changed
    assert ("is_owner", "added") in changed
    assert snapshot.lsp_status == "unavailable"
    assert snapshot.fallback_strategy == "python_ast+git_grep"
    assert snapshot.text_search_backend == "git-grep"


def test_search_repository_text_is_revision_bound(git_repo: Path):
    (git_repo / "auth.py").write_text("def check_role(role):\n    return role == 'admin'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth")
    head = run_git(git_repo, "rev-parse", "HEAD")

    matches = search_repository_text(git_repo, head, "check_role", max_results=5)

    assert matches[0].path == "auth.py"
    assert matches[0].line_number == 1
    assert "def check_role" in matches[0].line


def test_collect_python_symbols_tolerates_utf8_bom(git_repo: Path):
    (git_repo / "bom.py").write_text("\ufeffdef with_bom():\n    return True\n", encoding="utf-8")
    run_git(git_repo, "add", "bom.py")
    run_git(git_repo, "commit", "-m", "add bom file")
    head = run_git(git_repo, "rev-parse", "HEAD")

    symbols = collect_python_symbols(git_repo, head, paths=["bom.py"])

    assert [symbol.qualified_name for symbol in symbols] == ["with_bom"]


def test_repository_intelligence_summary_and_dict(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change add")
    head = run_git(git_repo, "rev-parse", "HEAD")

    snapshot = build_repository_intelligence(git_repo, base, head, changed_files=["app.py"])
    payload = repository_intelligence_to_dict(snapshot)
    summary = summarize_repository_intelligence(snapshot)

    assert payload["revision"] == head
    assert payload["base_revision"] == base
    assert "LSP unavailable; using python_ast+git_grep" in summary
    assert "modified function add app.py:1-2" in summary


def test_repository_intelligence_uses_optional_exact_cache_and_keeps_snapshot_authoritative(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch,
):
    (git_repo / "auth.py").write_text(
        "def check_role(role):\n    return role == 'admin'\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "base auth")
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text(
        "def check_role(role):\n    return role in {'admin', 'owner'}\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "head auth")
    head = run_git(git_repo, "rev-parse", "HEAD")
    backend = RepositoryKnowledgeCache(
        MemoryStore(tmp_path / "memory"),
        mode=MemoryMode.READ_WRITE,
        clock=lambda: "2026-07-14T00:00:00Z",
    )

    first = build_repository_intelligence(
        git_repo,
        base,
        head,
        changed_files=["auth.py"],
        cache_backend=backend,
        repository_key=_repository_key(git_repo),
        review_id="review-001",
    )
    assert first.cache_provenance is not None
    assert first.cache_provenance.status is RepositoryCacheStatus.MISS

    monkeypatch.setattr(
        "review_agent.repository_intelligence.detect_changed_symbols",
        lambda *args, **kwargs: pytest.fail("an exact cache hit reran AST analysis"),
    )
    hit = build_repository_intelligence(
        git_repo,
        base,
        head,
        changed_files=["auth.py"],
        cache_backend=backend,
        repository_key=_repository_key(git_repo),
        review_id="review-002",
    )

    assert hit.cache_provenance is not None
    assert hit.cache_provenance.status is RepositoryCacheStatus.HIT
    assert hit.base_revision == base
    assert hit.revision == head
    assert hit.changed_symbols == first.changed_symbols
    assert hit.cache_provenance.entry_id == first.cache_provenance.entry_id
    assert hit.cache_provenance.fallback == (
        ("fallback_strategy", "python_ast+git_grep"),
        ("lsp_status", "unavailable"),
        ("text_search_backend", "git-grep"),
    )


def test_repository_intelligence_cache_configuration_changes_miss(
    git_repo: Path,
    tmp_path: Path,
):
    base = run_git(git_repo, "rev-parse", "HEAD")
    run_git(git_repo, "commit", "--allow-empty", "-m", "empty head")
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = MemoryStore(tmp_path / "memory")
    repository_key = _repository_key(git_repo)
    backend = RepositoryKnowledgeCache(
        store,
        mode=MemoryMode.READ_WRITE,
        clock=lambda: "2026-07-14T00:00:00Z",
    )

    common = {
        "lsp_status": "unavailable",
        "fallback_strategy": "python_ast+git_grep",
        "text_search_backend": "git-grep",
        "analyzer_version": "repository-intelligence-v1",
        "python_ast_version": "3.12-v1",
        "text_search_backend_version": "git-2.45",
        "analysis_configuration": {"ignored_paths": []},
    }
    variants = (
        {"lsp_status": "available"},
        {"fallback_strategy": "lsp+python_ast+git_grep"},
        {"text_search_backend": "ripgrep"},
        {"analyzer_version": "repository-intelligence-v2"},
        {"python_ast_version": "3.12-v2"},
        {"text_search_backend_version": "git-2.46"},
        {"analysis_configuration": {"ignored_paths": ["vendor/**"]}},
    )
    snapshots = []
    for overrides in ({}, *variants):
        snapshots.append(
            build_repository_intelligence(
                git_repo,
                base,
                head,
                changed_files=[],
                cache_backend=backend,
                repository_key=repository_key,
                **{**common, **overrides},
            )
        )

    assert all(
        snapshot.cache_provenance.status is RepositoryCacheStatus.MISS
        for snapshot in snapshots
    )
    assert len({snapshot.cache_provenance.key_hash for snapshot in snapshots}) == len(
        snapshots
    )
    entries = store.list_knowledge_entries(repository_key)
    assert len(entries) == len(snapshots)
    assert len({entry.blob_hash for entry in entries}) == 1


def test_repository_intelligence_serialization_stays_session_compatible_with_cache_metadata(
    git_repo: Path,
):
    base = run_git(git_repo, "rev-parse", "HEAD")
    backend = RepositoryKnowledgeCache(None, mode=MemoryMode.OFF)
    snapshot = build_repository_intelligence(
        git_repo,
        base,
        base,
        changed_files=[],
        cache_backend=backend,
        repository_key=_repository_key(git_repo),
    )

    payload = repository_intelligence_to_dict(snapshot)

    assert snapshot.cache_provenance is not None
    assert snapshot.cache_provenance.status is RepositoryCacheStatus.OFF
    assert set(payload) == {
        "base_revision",
        "revision",
        "changed_symbols",
        "lsp_status",
        "fallback_strategy",
        "text_search_backend",
    }
    assert repository_intelligence_from_dict(payload) == snapshot


def test_repository_intelligence_cache_resolves_moving_revision_before_keying(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "moving.py").write_text(
        "def value():\n    return 1\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "moving.py")
    run_git(git_repo, "commit", "-m", "first moving head")
    first_head = run_git(git_repo, "rev-parse", "HEAD")
    backend = RepositoryKnowledgeCache(
        MemoryStore(tmp_path / "memory"),
        mode=MemoryMode.READ_WRITE,
        clock=lambda: "2026-07-14T00:00:00Z",
    )

    first = build_repository_intelligence(
        git_repo,
        base,
        "HEAD",
        changed_files=["moving.py"],
        cache_backend=backend,
        repository_key=_repository_key(git_repo),
    )

    (git_repo / "moving.py").write_text(
        "def value():\n    return 2\n\ndef added():\n    return True\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "moving.py")
    run_git(git_repo, "commit", "-m", "move symbolic head")
    second_head = run_git(git_repo, "rev-parse", "HEAD")
    second = build_repository_intelligence(
        git_repo,
        base,
        "HEAD",
        changed_files=["moving.py"],
        cache_backend=backend,
        repository_key=_repository_key(git_repo),
    )

    assert first.revision == first_head
    assert second.revision == second_head
    assert first.cache_provenance.status is RepositoryCacheStatus.MISS
    assert second.cache_provenance.status is RepositoryCacheStatus.MISS
    assert first.cache_provenance.key_hash != second.cache_provenance.key_hash
    assert first.cache_provenance.key.revision_binding == f"{base}..{first_head}"
    assert second.cache_provenance.key.revision_binding == f"{base}..{second_head}"


def test_repository_intelligence_rejects_mismatched_repository_namespace(
    git_repo: Path,
) -> None:
    head = run_git(git_repo, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="authorized repository identity"):
        build_repository_intelligence(
            git_repo,
            head,
            head,
            changed_files=[],
            cache_backend=RepositoryKnowledgeCache(None, mode=MemoryMode.OFF),
            repository_key="f" * 64,
        )
