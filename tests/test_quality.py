import os
from pathlib import Path
import sys

import pytest

from conftest import run_git
from review_agent.models import (
    Assignment,
    InitialContext,
    RiskAssessment,
    RiskLevel,
)
from review_agent.quality import (
    QualityGateDefinition,
    detect_quality_gates,
    discover_quality_gate_plan,
    quality_gate_policy_decision,
    run_python_compile_gate,
)
from review_agent.quality_runner import (
    _redact_output,
    _run_bounded_process,
    execute_quality_gate,
)


def test_detect_python_compile_for_python_repo(tmp_path: Path):
    (tmp_path / "app.py").write_text("print('hello')\n", encoding="utf-8")

    gates = detect_quality_gates(tmp_path)

    assert gates == ["python_compile"]


def test_python_compile_gate_passes_for_valid_python(tmp_path: Path):
    (tmp_path / "app.py").write_text("def ok():\n    return 1\n", encoding="utf-8")

    result = run_python_compile_gate(tmp_path)

    assert result.name == "python_compile"
    assert result.status == "passed"


def test_python_compile_gate_passes_for_utf8_bom_python(tmp_path: Path):
    (tmp_path / "app.py").write_text("\ufeffdef ok():\n    return 1\n", encoding="utf-8")

    result = run_python_compile_gate(tmp_path)

    assert result.status == "passed"


def test_python_compile_gate_fails_for_invalid_python(tmp_path: Path):
    (tmp_path / "bad.py").write_text("def broken(:\n    return 1\n", encoding="utf-8")

    result = run_python_compile_gate(tmp_path)

    assert result.status == "failed"
    assert "SyntaxError" in result.summary


def test_revision_python_compile_ignores_dirty_worktree(git_repo: Path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def broken(:\n", encoding="utf-8")

    gates = detect_quality_gates(git_repo, revision=head)
    result = run_python_compile_gate(git_repo, revision=head)

    assert gates == ["python_compile"]
    assert result.status == "passed"
    assert "Compiled 1 Python files" in result.summary


def test_revision_python_compile_reads_non_checked_out_commit(git_repo: Path):
    (git_repo / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    run_git(git_repo, "add", "bad.py")
    run_git(git_repo, "commit", "-m", "add invalid python target")
    target_head = run_git(git_repo, "rev-parse", "HEAD")

    (git_repo / "bad.py").write_text("def fixed():\n    return 1\n", encoding="utf-8")
    run_git(git_repo, "add", "bad.py")
    run_git(git_repo, "commit", "-m", "fix python on current head")

    result = run_python_compile_gate(git_repo, revision=target_head)

    assert result.status == "failed"
    assert "bad.py" in result.summary


def test_revision_python_compile_accepts_pep263_latin1(git_repo: Path):
    (git_repo / "latin.py").write_bytes(
        b"# coding: latin-1\nlabel = 'caf\xe9'\n"
    )
    run_git(git_repo, "add", "latin.py")
    run_git(git_repo, "commit", "-m", "add latin1 python")
    head = run_git(git_repo, "rev-parse", "HEAD")

    result = run_python_compile_gate(git_repo, revision=head)

    assert result.status == "passed"


def test_revision_python_compile_reports_unknown_encoding(git_repo: Path):
    (git_repo / "unknown_encoding.py").write_bytes(
        b"# coding: not-a-real-codec\nvalue = 1\n"
    )
    run_git(git_repo, "add", "unknown_encoding.py")
    run_git(git_repo, "commit", "-m", "add unknown encoding")
    head = run_git(git_repo, "rev-parse", "HEAD")

    result = run_python_compile_gate(git_repo, revision=head)

    assert result.status == "failed"
    assert "unknown_encoding.py" in result.summary
    assert "encoding" in result.summary.lower()


def test_worktree_python_compile_reports_invalid_encoded_bytes(tmp_path: Path):
    (tmp_path / "invalid_bytes.py").write_bytes(b"value = '\xff'\n")

    result = run_python_compile_gate(tmp_path)

    assert result.status == "failed"
    assert "invalid_bytes.py" in result.summary
    assert "encoding" in result.summary.lower()


def test_python_compile_file_limit_is_a_terminal_gate_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "one.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "two.py").write_text("value = 2\n", encoding="utf-8")
    monkeypatch.setattr("review_agent.quality.MAX_REVISION_FILES", 1)

    result = run_python_compile_gate(tmp_path)

    assert result.status == "error"
    assert result.reason is not None
    assert "exceeds file limit" in result.reason


def test_revision_discovers_configured_python_gates_and_ignores_dirty_config(
    git_repo: Path,
) -> None:
    (git_repo / "tests").mkdir()
    (git_repo / "tests" / "test_app.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )
    (git_repo / "pyproject.toml").write_text(
        """
[tool.ruff]
[tool.mypy]
[tool.pyright]
[tool.pytest.ini_options]

[[tool.review-agent.quality-gates]]
name = "bandit_security"
category = "security"
cost = "expensive"
command = ["python", "-m", "bandit", "-r", "."]
blocking = true
timeout_seconds = 45
trigger_risks = ["low", "medium", "high", "critical"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "pyproject.toml", "tests/test_app.py")
    run_git(git_repo, "commit", "-m", "configure quality gates")
    head = run_git(git_repo, "rev-parse", "HEAD")

    (git_repo / "pyproject.toml").write_text(
        "[tool.review-agent]\n",
        encoding="utf-8",
    )
    plan = discover_quality_gate_plan(git_repo, head)

    assert [gate.name for gate in plan.gates] == [
        "python_compile",
        "ruff",
        "mypy",
        "pyright",
        "pytest",
        "bandit_security",
    ]
    configured = plan.gates[-1]
    assert configured.source == "repository_config"
    assert configured.blocking is True
    assert configured.timeout_seconds == 45.0
    assert plan.discovery_issues == []


def test_invalid_repository_gate_is_reported_and_never_planned(
    git_repo: Path,
) -> None:
    (git_repo / "pyproject.toml").write_text(
        """
[[tool.review-agent.quality-gates]]
name = "unsafe_security"
command = ["powershell", "-Command", "Invoke-WebRequest example.test"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "pyproject.toml")
    run_git(git_repo, "commit", "-m", "add unsafe gate")
    head = run_git(git_repo, "rev-parse", "HEAD")

    plan = discover_quality_gate_plan(git_repo, head)

    assert "unsafe_security" not in [gate.name for gate in plan.gates]
    assert any("python -m module" in issue for issue in plan.discovery_issues)


def test_gate_definition_rejects_unapproved_modules_and_mutating_flags() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        QualityGateDefinition(
            name="pip_gate",
            category="security",
            cost="expensive",
            source="repository_config",
            command=["python", "-m", "pip", "install", "."],
        )
    with pytest.raises(ValueError, match="mutating"):
        QualityGateDefinition(
            name="ruff_fix",
            category="lint",
            cost="cheap",
            source="repository_config",
            command=["python", "-m", "ruff", "check", ".", "--fix"],
        )


def test_missing_gate_module_returns_auditable_unavailable_result(
    git_repo: Path,
) -> None:
    head = run_git(git_repo, "rev-parse", "HEAD")
    gate = QualityGateDefinition(
        name="ruff",
        category="lint",
        cost="cheap",
        source="builtin",
        command=["python", "-m", "ruff", "check", "."],
    )

    execution = execute_quality_gate(
        git_repo,
        head,
        gate,
        module_available=lambda _module: False,
    )

    assert execution.result.status == "unavailable"
    assert execution.result.reason is not None
    assert "not installed" in execution.raw_output


def test_command_gate_runs_only_committed_snapshot_with_network_denied(
    git_repo: Path,
) -> None:
    (git_repo / "tests").mkdir()
    test_file = git_repo / "tests" / "test_network.py"
    test_file.write_text(
        """
import socket
import unittest

class GateTest(unittest.TestCase):
    def test_network_is_denied(self):
        with self.assertRaises(PermissionError):
            socket.socket()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "tests/test_network.py")
    run_git(git_repo, "commit", "-m", "add isolated gate test")
    head = run_git(git_repo, "rev-parse", "HEAD")
    test_file.write_text(
        "import unittest\nclass Dirty(unittest.TestCase):\n"
        "    def test_dirty(self):\n        self.fail('dirty worktree leaked')\n",
        encoding="utf-8",
    )
    gate = QualityGateDefinition(
        name="unittest_gate",
        category="test",
        cost="expensive",
        source="repository_config",
        command=["python", "-m", "unittest", "discover", "-s", "tests"],
        trigger_risks=["high", "critical"],
    )

    execution = execute_quality_gate(git_repo, head, gate)

    assert execution.result.status == "passed"
    assert execution.result.exit_code == 0
    assert "OK" in execution.raw_output


def test_deep_gate_policy_uses_risk_and_security_reviewer() -> None:
    gate = QualityGateDefinition(
        name="bandit_security",
        category="security",
        cost="expensive",
        source="repository_config",
        command=["python", "-m", "bandit", "-r", "."],
        trigger_risks=["critical"],
    )
    low_risk = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={},
        reasons=[],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=[],
    )
    assignment = Assignment(
        role="Security Specialist Reviewer",
        mission="Inspect security",
        assignment_reason=["security review"],
        assigned_contract=["regression_safety"],
        required_checks=["inspect authorization"],
        initial_context=InitialContext(),
        max_turns=1,
        max_tool_calls=1,
    )

    should_run, reason = quality_gate_policy_decision(
        gate,
        low_risk,
        [assignment],
    )

    assert should_run is True
    assert "Security Specialist" in reason


def test_bounded_process_enforces_timeout_and_output_limit(tmp_path: Path) -> None:
    timeout = _run_bounded_process(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        cwd=tmp_path,
        env=dict(os.environ),
        timeout_seconds=0.05,
        max_output_bytes=1024,
        output_path=tmp_path / "timeout.bin",
    )
    limited = _run_bounded_process(
        [sys.executable, "-c", "print('x' * 4096)"],
        cwd=tmp_path,
        env=dict(os.environ),
        timeout_seconds=5,
        max_output_bytes=128,
        output_path=tmp_path / "limited.bin",
    )

    assert timeout.status == "timed_out"
    assert limited.status == "error"
    assert limited.output_truncated is True
    assert len(limited.output.encode("utf-8")) <= 128
    assert (tmp_path / "limited.bin").stat().st_size <= 128


@pytest.mark.parametrize(
    "raw",
    [
        "API_KEY=super-secret",
        "Authorization: Bearer abc.def.ghi",
        "password: hunter2",
    ],
)
def test_gate_output_redacts_common_secret_patterns(raw: str) -> None:
    redacted = _redact_output(raw)

    assert "[REDACTED]" in redacted
    assert "super-secret" not in redacted
    assert "abc.def.ghi" not in redacted
    assert "hunter2" not in redacted
