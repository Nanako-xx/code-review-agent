from pathlib import Path
import json

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


def test_tool_gateway_denies_tools_outside_allowlist_and_counts_attempt_without_observation(
    git_repo: Path,
    tmp_path: Path,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path / "allowed-tools")
    gateway = ToolGateway(
        git_repo,
        base_revision=head,
        head_revision=head,
        observation_store=store,
        allowed_tools=("read_range",),
    )

    with pytest.raises(ToolGatewayError, match="not allowed") as caught:
        gateway.execute("compare_base_head", {"path": "app.py"})

    assert caught.value.code == "tool_not_allowed"
    assert caught.value.tool_name == "compare_base_head"
    assert gateway.attempted_tool_calls == 1
    assert gateway.denied_tool_calls == 1
    assert store.list_observations() == []

    gateway.execute(
        "read_range",
        {"path": "app.py", "revision": "head", "line_start": 1, "line_end": 1},
    )
    assert gateway.attempted_tool_calls == 2
    assert gateway.denied_tool_calls == 1
    assert len(store.list_observations()) == 1


def test_tool_gateway_rejects_unknown_constructor_allowlist_item(
    git_repo: Path,
    tmp_path: Path,
):
    head = run_git(git_repo, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="unsupported allowed tool"):
        ToolGateway(
            git_repo,
            base_revision=head,
            head_revision=head,
            observation_store=ObservationStore(tmp_path / "unknown-tool"),
            allowed_tools=("write_file",),
        )


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


def test_tool_gateway_read_commit_messages_is_bounded_to_fixed_revision_range(
    git_repo: Path,
    tmp_path: Path,
):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return int(a) + int(b)\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(
        git_repo,
        "commit",
        "-m",
        "preserve integer addition",
        "-m",
        "Adds the documented acceptance behavior.",
        "-m",
        "Requirement: INT-42",
    )
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path / "commit-observations")
    gateway = ToolGateway(
        git_repo,
        base_revision=base,
        head_revision=head,
        observation_store=store,
        max_commit_messages=1,
    )

    result = gateway.execute(
        "read_commit_messages",
        {
            "base_revision": base,
            "head_revision": head,
            "revision_range": f"{base}..{head}",
            "max_commits": 100,
        },
    )

    messages = json.loads(result.context_view)
    assert len(messages) == 1
    assert messages[0]["hash"] == head
    assert messages[0]["subject"] == "preserve integer addition"
    assert messages[0]["body"] == "Adds the documented acceptance behavior."
    assert messages[0]["trailers"] == [
        {"key": "Requirement", "value": "INT-42"}
    ]
    observation = store.list_observations()[0]
    assert observation.source == "git.read_commit_messages"
    assert observation.revision == f"{base}..{head}"
    assert observation.path is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"base_revision": "HEAD~9"},
        {"head_revision": "main"},
        {"revision": "HEAD~2..HEAD"},
        {"revision_range": "main..HEAD"},
    ],
)
def test_tool_gateway_read_commit_messages_rejects_revision_override(
    git_repo: Path,
    tmp_path: Path,
    arguments,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    gateway = ToolGateway(
        git_repo,
        base_revision=head,
        head_revision=head,
        observation_store=ObservationStore(tmp_path / "commit-binding"),
    )

    with pytest.raises(ToolGatewayError, match="unauthorized revision binding"):
        gateway.execute("read_commit_messages", arguments)
