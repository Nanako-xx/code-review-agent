from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

import pytest

from review_agent.diff_artifact import DiffArtifactIndex, DiffFileIndex
from review_agent.pr_workspace import PRMetadata, PRWorkspaceStore
from review_agent.review_protocol import (
    IntentPacket,
    IntentSource,
    RiskDecision,
    RiskLevel,
)
from review_agent.revision import RepositoryIdentity
from review_agent.risk_runtime import (
    RiskRecord,
    RiskRuntime,
    compile_risk_record,
    deterministic_risk_floor,
)


BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40


def _intent(source: IntentSource | None) -> IntentPacket:
    if source is None:
        return IntentPacket(
            goal=None,
            source=None,
            uncertainties=("The PR has no reliable goal statement.",),
        )
    return IntentPacket(
        goal="Preserve request behavior.",
        source=source,
        uncertainties=(),
    )


@pytest.mark.parametrize(
    ("changed_file_count", "source", "expected"),
    [
        (100, IntentSource.EXPLICIT, RiskLevel.LOW),
        (101, IntentSource.EXPLICIT, RiskLevel.MEDIUM),
        (1, IntentSource.INFERRED, RiskLevel.MEDIUM),
        (1, None, RiskLevel.HIGH),
        (101, None, RiskLevel.HIGH),
    ],
)
def test_deterministic_risk_floor_has_exact_file_and_intent_boundaries(
    changed_file_count: int,
    source: IntentSource | None,
    expected: RiskLevel,
) -> None:
    assert deterministic_risk_floor(changed_file_count, _intent(source)) is expected


def test_risk_final_level_is_the_maximum_of_all_floors_and_model_level() -> None:
    intent = _intent(IntentSource.EXPLICIT)

    model_raises = compile_risk_record(
        snapshot_id="S-" + "a" * 64,
        changed_file_count=2,
        intent=intent,
        model_decision=RiskDecision(RiskLevel.HIGH),
    )
    deterministic_floor_wins = compile_risk_record(
        snapshot_id="S-" + "b" * 64,
        changed_file_count=2,
        intent=intent,
        model_decision=RiskDecision(RiskLevel.LOW),
        additional_floors=(RiskLevel.CRITICAL,),
    )

    assert model_raises.final_level is RiskLevel.HIGH
    assert deterministic_floor_wins.deterministic_floor is RiskLevel.CRITICAL
    assert deterministic_floor_wins.final_level is RiskLevel.CRITICAL


def test_risk_record_and_downstream_projection_are_minimal() -> None:
    record = compile_risk_record(
        snapshot_id="S-" + "c" * 64,
        changed_file_count=3,
        intent=_intent(IntentSource.EXPLICIT),
        model_decision=RiskDecision(RiskLevel.MEDIUM),
    )

    assert [field.name for field in fields(RiskRecord)] == [
        "snapshot_id",
        "deterministic_floor",
        "model_level",
        "final_level",
    ]
    assert record.to_dict() == {
        "schema_version": "risk_decision_v2",
        "snapshot_id": "S-" + "c" * 64,
        "deterministic_floor": "low",
        "model_level": "medium",
        "final_level": "medium",
    }
    assert record.to_decision() == RiskDecision(RiskLevel.MEDIUM)
    for forbidden in (
        "dimensions",
        "reasons",
        "signal_refs",
        "uncertainties",
        "suggested_focus",
    ):
        assert forbidden not in record.to_dict()
        assert forbidden not in record.to_decision().to_dict()


def _workspace(tmp_path: Path):
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    git_common.mkdir(parents=True)
    identity = RepositoryIdentity(
        canonical_path=str(repository.resolve()),
        git_common_dir=str(git_common.resolve()),
        origin_url=None,
    )
    store = PRWorkspaceStore(tmp_path / "ra")
    workspace = store.create_or_load_workspace(
        store.resolve_pr(identity, "local", "risk-task"),
        PRMetadata(title="Risk task"),
    )
    snapshot = store.create_or_load_snapshot(workspace, BASE_SHA, HEAD_SHA)
    return store, snapshot


def _diff_index(snapshot_id: str, file_count: int) -> DiffArtifactIndex:
    files = tuple(
        DiffFileIndex(
            file_index=index,
            path=f"src/file_{index:03d}.py",
            previous_path=None,
            status="modify",
            additions=1,
            deletions=0,
            binary=False,
            submodule=False,
            byte_start=index * 2,
            byte_end=index * 2 + 1,
            hunks=(),
        )
        for index in range(file_count)
    )
    return DiffArtifactIndex(
        snapshot_id=snapshot_id,
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        patch_artifact_id="A-" + "d" * 64,
        diff_sha256="e" * 64,
        diff_size_bytes=max(1, file_count * 2),
        files=files,
    )


def test_risk_runtime_persists_one_snapshot_bound_minimal_record(
    tmp_path: Path,
) -> None:
    store, snapshot = _workspace(tmp_path)
    runtime = RiskRuntime(store)

    record = runtime.finalize(
        snapshot,
        _diff_index(snapshot.snapshot_id, 101),
        _intent(IntentSource.EXPLICIT),
        model_decision=RiskDecision(RiskLevel.LOW),
    )

    assert record.final_level is RiskLevel.MEDIUM
    assert runtime.load(snapshot) == record
    persisted = json.loads(
        (snapshot.path / "Risk" / "risk.json").read_text("utf-8")
    )
    assert persisted == record.to_dict()


def test_risk_runtime_rejects_a_diff_from_another_snapshot(tmp_path: Path) -> None:
    store, snapshot = _workspace(tmp_path)

    with pytest.raises(ValueError, match="Snapshot"):
        RiskRuntime(store).finalize(
            snapshot,
            _diff_index("S-" + "f" * 64, 1),
            _intent(IntentSource.EXPLICIT),
        )
