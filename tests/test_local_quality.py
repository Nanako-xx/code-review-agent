from __future__ import annotations

from pathlib import Path

import pytest

from review_agent.artifacts import artifact_schema
from review_agent.local_quality import (
    CommandExecution,
    LocalQualityCommand,
    LocalQualityPlan,
    LocalQualityRunner,
    QualityGateStatus,
)


SNAPSHOT_ID = "S-" + "a" * 64


class RecordingExecutor:
    def __init__(self, executions: list[CommandExecution]) -> None:
        self.executions = list(executions)
        self.calls: list[tuple[Path, tuple[str, ...], float]] = []

    def run(
        self,
        repository: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
    ) -> CommandExecution:
        self.calls.append((repository, argv, timeout_seconds))
        return self.executions.pop(0)


class MemorySink:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, bytes, str]] = []

    def publish(
        self,
        *,
        snapshot_id: str,
        logical_name: str,
        content: bytes,
        content_type: str,
    ) -> str:
        self.items.append((snapshot_id, logical_name, content, content_type))
        return "A-" + f"{len(self.items):064x}"


def _command(name: str, *, timeout_seconds: float = 30) -> LocalQualityCommand:
    return LocalQualityCommand(
        name=name,
        kind="lint",
        argv=("ruff", "check", "."),
        timeout_seconds=timeout_seconds,
    )


def test_quality_gate_has_only_four_product_statuses() -> None:
    assert tuple(status.value for status in QualityGateStatus) == (
        "passed",
        "failed",
        "unavailable",
        "error",
    )
    assert "skipped" not in {status.value for status in QualityGateStatus}
    assert "timed_out" not in {status.value for status in QualityGateStatus}
    assert artifact_schema("quality_gate_v2") == "quality_gate_result_v2"
    assert artifact_schema("changed_symbols_v2") == "changed_symbols_v2"


def test_nonzero_static_check_is_failed_and_does_not_stop_later_checks(
    tmp_path: Path,
) -> None:
    executor = RecordingExecutor(
        [
            CommandExecution(exit_code=1, stdout=b"name is undefined\n", duration_seconds=1),
            CommandExecution(exit_code=0, stdout=b"ok\n", duration_seconds=2),
        ]
    )
    runner = LocalQualityRunner(executor=executor)
    plan = LocalQualityPlan(commands=(_command("lint"), _command("types")))

    result = runner.run(tmp_path, SNAPSHOT_ID, plan, MemorySink())

    assert result.status is QualityGateStatus.FAILED
    assert [item.status for item in result.commands] == [
        QualityGateStatus.FAILED,
        QualityGateStatus.PASSED,
    ]
    assert result.commands[0].reason_code == "static_check_failed"
    assert len(executor.calls) == 2


def test_unavailable_tool_and_timeout_use_unavailable_and_error() -> None:
    executor = RecordingExecutor(
        [
            CommandExecution(
                exit_code=None,
                unavailable=True,
                error_code="tool_not_found",
                duration_seconds=0,
            ),
            CommandExecution(
                exit_code=None,
                timed_out=True,
                error_code="command_timeout",
                duration_seconds=10,
            ),
        ]
    )
    runner = LocalQualityRunner(executor=executor)
    plan = LocalQualityPlan(commands=(_command("missing"), _command("slow")))

    result = runner.run(Path.cwd(), SNAPSHOT_ID, plan, MemorySink())

    assert [item.status for item in result.commands] == [
        QualityGateStatus.UNAVAILABLE,
        QualityGateStatus.ERROR,
    ]
    assert result.status is QualityGateStatus.ERROR
    assert result.commands[1].reason_code == "command_timeout"


def test_empty_plan_is_unavailable_without_executing_anything(tmp_path: Path) -> None:
    executor = RecordingExecutor([])

    result = LocalQualityRunner(executor=executor).run(
        tmp_path,
        SNAPSHOT_ID,
        LocalQualityPlan(commands=()),
        MemorySink(),
    )

    assert result.status is QualityGateStatus.UNAVAILABLE
    assert result.commands == ()
    assert executor.calls == []


def test_command_and_stage_watchdogs_are_bounded_to_1800_seconds() -> None:
    with pytest.raises(ValueError, match="1800"):
        _command("too-long", timeout_seconds=1801)
    with pytest.raises(ValueError, match="1800"):
        LocalQualityPlan(commands=(), total_timeout_seconds=1801)

    executor = RecordingExecutor(
        [CommandExecution(exit_code=0, duration_seconds=1)]
    )
    runner = LocalQualityRunner(executor=executor)
    runner.run(
        Path.cwd(),
        SNAPSHOT_ID,
        LocalQualityPlan(
            commands=(_command("bounded", timeout_seconds=1700),),
            total_timeout_seconds=12,
        ),
        MemorySink(),
    )
    assert executor.calls[0][2] == 12


def test_large_stdout_is_persisted_whole_with_only_a_small_preview(
    tmp_path: Path,
) -> None:
    output = b"x" * 60_000
    executor = RecordingExecutor(
        [CommandExecution(exit_code=0, stdout=output, duration_seconds=1)]
    )
    sink = MemorySink()

    result = LocalQualityRunner(executor=executor).run(
        tmp_path,
        SNAPSHOT_ID,
        LocalQualityPlan(commands=(_command("large"),)),
        sink,
    )

    command = result.commands[0]
    assert sink.items == [
        (SNAPSHOT_ID, "quality/large/stdout.log", output, "text/plain")
    ]
    assert command.stdout_artifact_id == "A-" + f"{1:064x}"
    assert len(command.stdout_preview.encode("utf-8")) <= 2048
    assert command.stdout_truncated is True


@pytest.mark.parametrize(
    "argv",
    [
        ("curl", "https://example.test"),
        ("wget", "https://example.test"),
        ("pip", "install", "ruff"),
        ("python", "-m", "pip", "install", "ruff"),
        ("npm", "install"),
        ("npx", "eslint", "."),
    ],
)
def test_quality_configuration_rejects_network_or_dependency_install_commands(
    argv: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="network|install|download"):
        LocalQualityCommand(
            name="unsafe",
            kind="lint",
            argv=argv,
            timeout_seconds=30,
        )
