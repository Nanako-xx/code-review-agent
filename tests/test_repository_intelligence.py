from pathlib import Path

from conftest import run_git
from review_agent.repository_intelligence import (
    build_repository_intelligence,
    collect_python_symbols,
    repository_intelligence_to_dict,
    search_repository_text,
    summarize_repository_intelligence,
)


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
