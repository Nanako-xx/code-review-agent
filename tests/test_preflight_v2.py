from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from review_agent.local_quality import (
    LocalQualityPlan,
    QualityGateResult,
    QualityGateStatus,
)
from review_agent.preflight import (
    DeterministicPreflight,
    PreflightBlockedError,
)
from review_agent.repository_intelligence import ChangedSymbolsV2


SNAPSHOT_ID = "S-" + "a" * 64


class FakePublisher:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.counter = 0

    def publish_create_only(self, snapshot, relative_path: str, content: bytes):
        self.counter += 1
        return SimpleNamespace(artifact_id="A-" + f"{self.counter:064x}")


class FakeDiffStore:
    def __init__(self, events: list[str], *, fail: bool = False, wrong_snapshot: bool = False) -> None:
        self.events = events
        self.fail = fail
        self.wrong_snapshot = wrong_snapshot

    def materialize(self, repository: Path, snapshot):
        self.events.append("diff")
        if self.fail:
            raise ValueError("diff unavailable")
        indexed_snapshot = "S-" + "f" * 64 if self.wrong_snapshot else snapshot.snapshot_id
        return SimpleNamespace(
            patch=SimpleNamespace(artifact_id="A-" + "1" * 64),
            index_artifact=SimpleNamespace(artifact_id="A-" + "2" * 64),
            index=SimpleNamespace(
                snapshot_id=indexed_snapshot,
                files=(SimpleNamespace(path="src/app.py"),),
            ),
        )


class FakeQualityRunner:
    def __init__(
        self,
        events: list[str],
        *,
        status: QualityGateStatus = QualityGateStatus.PASSED,
        fail: bool = False,
    ) -> None:
        self.events = events
        self.status = status
        self.fail = fail

    def run(self, repository, snapshot_id, plan, sink):
        self.events.append("quality")
        if self.fail:
            raise RuntimeError("runner crashed")
        return QualityGateResult(
            snapshot_id=snapshot_id,
            status=self.status,
            commands=(),
        )


def _changed_symbols_builder(events: list[str]):
    def build(repository, snapshot_id, base_sha, head_sha, changed_files):
        events.append("symbols")
        return ChangedSymbolsV2.empty(
            snapshot_id=snapshot_id,
            base_sha=base_sha,
            head_sha=head_sha,
            changed_files=changed_files,
        )

    return build


def _snapshot():
    return SimpleNamespace(
        snapshot_id=SNAPSHOT_ID,
        base_sha="b" * 40,
        head_sha="c" * 40,
    )


def test_preflight_order_is_diff_then_quality_then_changed_symbols() -> None:
    events: list[str] = []
    runner = DeterministicPreflight(
        workspace_store=FakePublisher(events),
        diff_store=FakeDiffStore(events),
        quality_runner=FakeQualityRunner(events),
        changed_symbols_builder=_changed_symbols_builder(events),
    )

    result = runner.run(
        Path.cwd(),
        _snapshot(),
        LocalQualityPlan(commands=()),
        sink=SimpleNamespace(),
    )

    assert events[:3] == ["diff", "quality", "symbols"]
    assert result.snapshot_id == SNAPSHOT_ID
    assert result.quality.snapshot_id == SNAPSHOT_ID
    assert result.changed_symbols.snapshot_id == SNAPSHOT_ID
    assert result.diff_artifact_id == "A-" + "1" * 64


@pytest.mark.parametrize(
    "status",
    [
        QualityGateStatus.FAILED,
        QualityGateStatus.UNAVAILABLE,
        QualityGateStatus.ERROR,
    ],
)
def test_nonpassing_quality_gate_does_not_block_changed_symbols(
    status: QualityGateStatus,
) -> None:
    events: list[str] = []
    runner = DeterministicPreflight(
        workspace_store=FakePublisher(events),
        diff_store=FakeDiffStore(events),
        quality_runner=FakeQualityRunner(events, status=status),
        changed_symbols_builder=_changed_symbols_builder(events),
    )

    result = runner.run(
        Path.cwd(),
        _snapshot(),
        LocalQualityPlan(commands=()),
        sink=SimpleNamespace(),
    )

    assert events[:3] == ["diff", "quality", "symbols"]
    assert result.quality.status is status


def test_quality_runtime_exception_is_recorded_as_error_and_symbols_continue() -> None:
    events: list[str] = []
    runner = DeterministicPreflight(
        workspace_store=FakePublisher(events),
        diff_store=FakeDiffStore(events),
        quality_runner=FakeQualityRunner(events, fail=True),
        changed_symbols_builder=_changed_symbols_builder(events),
    )

    result = runner.run(
        Path.cwd(),
        _snapshot(),
        LocalQualityPlan(commands=()),
        sink=SimpleNamespace(),
    )

    assert events[:3] == ["diff", "quality", "symbols"]
    assert result.quality.status is QualityGateStatus.ERROR
    assert result.quality.reason_code == "quality_runtime_error"


def test_changed_symbol_runtime_exception_is_an_explicit_nonblocking_coverage_error() -> None:
    events: list[str] = []

    def failing_builder(repository, snapshot_id, base_sha, head_sha, changed_files):
        events.append("symbols")
        raise RuntimeError("analyzer crashed")

    runner = DeterministicPreflight(
        workspace_store=FakePublisher(events),
        diff_store=FakeDiffStore(events),
        quality_runner=FakeQualityRunner(events),
        changed_symbols_builder=failing_builder,
    )

    result = runner.run(
        Path.cwd(),
        _snapshot(),
        LocalQualityPlan(commands=()),
        sink=SimpleNamespace(),
    )

    assert events[:3] == ["diff", "quality", "symbols"]
    assert result.changed_symbols.language_coverage[0].status == "error"
    assert (
        result.changed_symbols.language_coverage[0].reason_code
        == "analyzer_runtime_error"
    )


@pytest.mark.parametrize("wrong_snapshot", [False, True])
def test_diff_failure_or_snapshot_mismatch_blocks_preflight(
    wrong_snapshot: bool,
) -> None:
    events: list[str] = []
    runner = DeterministicPreflight(
        workspace_store=FakePublisher(events),
        diff_store=FakeDiffStore(
            events,
            fail=not wrong_snapshot,
            wrong_snapshot=wrong_snapshot,
        ),
        quality_runner=FakeQualityRunner(events),
        changed_symbols_builder=_changed_symbols_builder(events),
    )

    with pytest.raises(PreflightBlockedError, match="Diff|Snapshot"):
        runner.run(
            Path.cwd(),
            _snapshot(),
            LocalQualityPlan(commands=()),
            sink=SimpleNamespace(),
        )

    assert events == ["diff"]
