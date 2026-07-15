from pathlib import Path
import json

import pytest

from conftest import run_git
from review_agent.memory_models import MemoryScope
from review_agent.memory_retrieval import (
    MemoryQuery,
    RetrievalLimits,
    SnapshotMemoryQueryService,
)
from review_agent.observations import ObservationStore
from review_agent.tool_gateway import ToolGateway, ToolGatewayError
from tests.test_context import _combined_memory_snapshot, _memory_snapshot


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


def test_tool_gateway_query_project_memory_uses_only_bound_snapshot_and_records_observation(
    git_repo: Path,
    tmp_path: Path,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    snapshot = _memory_snapshot(head=head)
    service = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="assignment-memory",
        assignment_scope=MemoryScope(paths=("app.py",)),
    )
    store = ObservationStore(tmp_path / "memory-query")
    gateway = ToolGateway(
        git_repo,
        base_revision=head,
        head_revision="HEAD",
        observation_store=store,
        allowed_tools=("query_project_memory",),
        memory_query_service=service,
    )

    result = gateway.execute(
        "query_project_memory",
        {
            "assignment_id": "assignment-memory",
            "path": "app.py",
            "query": "approved rule",
        },
    )

    assert result.observation_ids
    assert gateway.memory_snapshot.snapshot_id in result.context_view
    assert "approved reviewer rule" in result.context_view
    payload = json.loads(result.context_view)
    assert payload["byte_size"] == len(result.context_view.encode("utf-8"))
    observation = store.list_observations()[0]
    assert observation.source == "memory.query_project_memory"
    assert observation.revision == f"head@{head}"
    assert gateway.attempted_tool_calls == 1


def test_tool_gateway_query_project_memory_rejects_unbounded_arguments_and_local_only_output(
    git_repo: Path,
    tmp_path: Path,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    snapshot = _memory_snapshot(head=head, local_only=True)
    service = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="assignment-memory",
        assignment_scope=MemoryScope(paths=("app.py",)),
    )
    store = ObservationStore(tmp_path / "memory-query-local")
    gateway = ToolGateway(
        git_repo,
        base_revision=head,
        head_revision=head,
        observation_store=store,
        allowed_tools=("query_project_memory",),
        memory_query_service=service,
    )

    with pytest.raises(ToolGatewayError, match="unsupported argument"):
        gateway.execute(
            "query_project_memory",
            {"assignment_id": "assignment-memory", "path": "app.py", "store": "live"},
        )

    result = gateway.execute(
        "query_project_memory",
        {"assignment_id": "assignment-memory", "path": "app.py"},
    )
    assert "approved reviewer rule" not in result.context_view
    assert snapshot.eligible_records[0].memory_id not in result.context_view
    assert len(store.list_observations()) == 1


def test_tool_gateway_rebuilds_preconstructed_service_to_expected_assignment_scope(
    git_repo: Path,
    tmp_path: Path,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    snapshot = _memory_snapshot(head=head)
    stale_service = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="assignment-memory",
        assignment_scope=MemoryScope(paths=("other.py",)),
    )
    gateway = ToolGateway(
        git_repo,
        base_revision=head,
        head_revision=head,
        observation_store=ObservationStore(tmp_path / "memory-rebound"),
        allowed_tools=("query_project_memory",),
        memory_query_service=stale_service,
        assignment_id="assignment-memory",
        assignment_scope=MemoryScope(paths=("app.py",)),
    )

    result = gateway.execute(
        "query_project_memory",
        {"assignment_id": "assignment-memory", "path": "app.py"},
    )

    assert "approved reviewer rule" in result.context_view
    assert gateway.memory_assignment_scope == MemoryScope(paths=("app.py",))
    assert stale_service.call_count == 0


def test_tool_gateway_local_only_records_cannot_perturb_remote_tool_payload(
    git_repo: Path,
    tmp_path: Path,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    normal = _memory_snapshot(head=head)
    local_only = _memory_snapshot(head=head, local_only=True)
    mixed = _combined_memory_snapshot(normal, local_only, memory_generation=888)

    def execute(snapshot, suffix):
        service = SnapshotMemoryQueryService(
            snapshot,
            assignment_id="assignment-memory",
            assignment_scope=MemoryScope(paths=("app.py",)),
        )
        gateway = ToolGateway(
            git_repo,
            base_revision=head,
            head_revision="HEAD",
            observation_store=ObservationStore(tmp_path / suffix),
            allowed_tools=("query_project_memory",),
            memory_query_service=service,
        )
        return gateway.execute(
            "query_project_memory",
            {"assignment_id": "assignment-memory", "path": "app.py"},
        ).context_view

    visible_payload = execute(normal, "visible-only")
    mixed_payload = execute(mixed, "mixed-local")

    assert mixed_payload == visible_payload
    assert local_only.eligible_records[0].memory_id not in mixed_payload
    assert mixed.snapshot_id not in mixed_payload


def test_memory_tool_rechecks_final_utf8_bytes_including_omitted_metadata(
    git_repo: Path,
    tmp_path: Path,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    snapshot = _memory_snapshot(
        head=head,
        statement="审查规则必须验证边界" * 30,
    )
    query = MemoryQuery(
        assignment_id="assignment-memory",
        path="app.py",
    )
    probe = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="assignment-memory",
        assignment_scope=MemoryScope(paths=("app.py",)),
    ).query(query)
    limits = RetrievalLimits(max_query_bytes=probe.byte_size)
    gateway = ToolGateway(
        git_repo,
        base_revision=head,
        head_revision=head,
        observation_store=ObservationStore(tmp_path / "memory-utf8-final"),
        allowed_tools=("query_project_memory",),
        memory_query_service=SnapshotMemoryQueryService(
            snapshot,
            assignment_id="assignment-memory",
            assignment_scope=MemoryScope(paths=("app.py",)),
            limits=limits,
        ),
        max_context_chars=limits.max_query_bytes,
    )

    result = gateway.execute(
        "query_project_memory",
        {"assignment_id": "assignment-memory", "path": "app.py"},
    )
    payload = json.loads(result.context_view)

    assert payload["byte_size"] == len(result.context_view.encode("utf-8"))
    assert payload["byte_size"] <= limits.max_query_bytes
    assert payload["records"] == []
    assert payload["omitted_memory_ids"] == [snapshot.eligible_records[0].memory_id]


def test_memory_gateway_rejects_snapshot_that_does_not_match_resolved_head(
    git_repo: Path,
    tmp_path: Path,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    snapshot = _memory_snapshot(head="f" * 40)
    service = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="assignment-memory",
        assignment_scope=MemoryScope(paths=("app.py",)),
    )

    with pytest.raises(ValueError, match="does not match"):
        ToolGateway(
            git_repo,
            base_revision=head,
            head_revision="HEAD",
            observation_store=ObservationStore(tmp_path / "memory-wrong-head"),
            memory_query_service=service,
        )
