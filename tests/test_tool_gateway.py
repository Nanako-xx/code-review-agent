from pathlib import Path

import pytest

from conftest import run_git
from review_agent.observations import ObservationStore
from review_agent.tool_gateway import ToolGateway, ToolGatewayError


def test_tool_gateway_read_range_records_observation(git_repo: Path, tmp_path: Path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path)
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=store)

    result = gateway.execute(
        "read_range",
        {"path": "app.py", "revision": "head", "line_start": 1, "line_end": 2},
    )

    assert len(result.observation_ids) == 1
    assert "def add" in result.context_view
    observation = store.list_observations()[0]
    assert observation.source == "git.read_range"
    assert observation.revision == f"head@{head}"
    assert observation.path == "app.py"
    assert observation.line_start == 1
    assert observation.line_end == 2


def test_tool_gateway_compare_base_head_records_diff(git_repo: Path, tmp_path: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change behavior")
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path)
    gateway = ToolGateway(git_repo, base_revision=base, head_revision=head, observation_store=store)

    result = gateway.execute("compare_base_head", {"path": "app.py"})

    assert len(result.observation_ids) == 1
    assert "-    return a + b" in result.context_view
    assert "+    return a - b" in result.context_view
    assert store.list_observations()[0].revision == f"{base}..{head}"


def test_tool_gateway_search_code_records_matches(git_repo: Path, tmp_path: Path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path)
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=store)

    result = gateway.execute("search_code", {"query": "def add", "revision": "head", "max_results": 5})

    assert len(result.observation_ids) == 1
    assert "app.py:1:def add" in result.context_view
    assert store.list_observations()[0].source == "git.search_code"


def test_tool_gateway_rejects_unsafe_paths(git_repo: Path, tmp_path: Path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=ObservationStore(tmp_path))

    with pytest.raises(ToolGatewayError, match="unsafe repository path"):
        gateway.execute("read_range", {"path": "../secret.txt", "revision": "head", "line_start": 1, "line_end": 1})

    with pytest.raises(ToolGatewayError, match="unsafe repository path"):
        gateway.execute("read_range", {"path": ".git/config", "revision": "head", "line_start": 1, "line_end": 1})


def test_tool_gateway_rejects_unauthorized_revision(git_repo: Path, tmp_path: Path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=ObservationStore(tmp_path))

    with pytest.raises(ToolGatewayError, match="unauthorized revision"):
        gateway.execute("read_range", {"path": "app.py", "revision": "main", "line_start": 1, "line_end": 1})


def test_tool_gateway_list_symbols_records_ast_observation(git_repo: Path, tmp_path: Path):
    (git_repo / "auth.py").write_text("def check_role(role):\n    return role == 'admin'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add symbol")
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path)
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=store)

    result = gateway.execute("list_symbols", {"path": "auth.py", "revision": "head"})

    assert len(result.observation_ids) == 1
    assert "function check_role auth.py:1-2" in result.context_view
    assert store.list_observations()[0].source == "repo_intelligence.list_symbols"


def test_tool_gateway_inspect_symbol_records_calls(git_repo: Path, tmp_path: Path):
    (git_repo / "auth.py").write_text(
        "def is_admin(user):\n"
        "    return check_role(user.role)\n\n"
        "def check_role(role):\n"
        "    return role == 'admin'\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add symbol calls")
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path)
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=store)

    result = gateway.execute("inspect_symbol", {"name": "is_admin", "revision": "head"})

    assert "is_admin" in result.context_view
    assert "calls: check_role" in result.context_view
    assert store.list_observations()[0].source == "repo_intelligence.inspect_symbol"


def test_tool_gateway_find_references_uses_revision_bound_text_search(git_repo: Path, tmp_path: Path):
    (git_repo / "auth.py").write_text("def check_role(role):\n    return role == 'admin'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add references")
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path)
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=store)

    result = gateway.execute("find_references", {"name": "check_role", "revision": "head", "max_results": 5})

    assert "auth.py:1:def check_role" in result.context_view
    assert store.list_observations()[0].source == "repo_intelligence.find_references"
