from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from review_agent.diff_artifact import (
    DiffArtifactIndex,
    DiffFileIndex,
    DiffHunkIndex,
)
from review_agent.portfolio import (
    AssignmentPlannerDraft,
    AssignmentPlannerProposal,
)
from review_agent.pr_workspace import PRMetadata, PRWorkspaceStore
from review_agent.repository_intelligence import (
    ChangedSymbolV2,
    ChangedSymbolsV2,
)
from review_agent.review_planning import (
    READ_ONLY_REVIEW_PERMISSIONS,
    ReviewPlanningRuntime,
    compile_review_plan,
    fixed_reviewer_slots,
)
from review_agent.review_protocol import (
    AssignmentTargets,
    IntentSource,
    ReviewPlan,
    ReviewerAssignment,
    ReviewerRoleKind,
    RiskLevel,
)
from review_agent.revision import RepositoryIdentity
from review_agent.risk_runtime import compile_risk_record
from review_agent.review_protocol import IntentPacket


BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
SNAPSHOT_ID = "S-" + "c" * 64


@pytest.mark.parametrize(
    ("level", "roles"),
    [
        (RiskLevel.LOW, (ReviewerRoleKind.CORE,)),
        (
            RiskLevel.MEDIUM,
            (ReviewerRoleKind.CORE, ReviewerRoleKind.ADVERSARIAL),
        ),
        (
            RiskLevel.HIGH,
            (
                ReviewerRoleKind.CORE,
                ReviewerRoleKind.ADVERSARIAL,
                ReviewerRoleKind.DYNAMIC,
            ),
        ),
        (
            RiskLevel.CRITICAL,
            (
                ReviewerRoleKind.CORE,
                ReviewerRoleKind.ADVERSARIAL,
                ReviewerRoleKind.DYNAMIC,
                ReviewerRoleKind.DYNAMIC,
            ),
        ),
    ],
)
def test_risk_has_a_fixed_reviewer_slot_mapping(
    level: RiskLevel,
    roles: tuple[ReviewerRoleKind, ...],
) -> None:
    slots = fixed_reviewer_slots(level)

    assert tuple(slot.role_kind for slot in slots) == roles
    assert len({slot.slot_id for slot in slots}) == len(slots)


def _draft(
    slot_id: str,
    *,
    perspective: str | None = None,
    files: tuple[str, ...] = ("src/api.py",),
) -> AssignmentPlannerDraft:
    return AssignmentPlannerDraft(
        slot_id=slot_id,
        perspective=perspective,
        mission=f"Review {slot_id} concerns in the changed request path.",
        targets=AssignmentTargets(
            files=files,
            symbols=(),
            hunks=(),
        ),
        checks=("Verify the changed request behavior.",),
    )


def test_planner_can_fill_tasks_but_cannot_change_slots_roles_or_permissions() -> None:
    proposal = AssignmentPlannerProposal(
        assignments=(
            _draft("core-1", perspective="public-api"),
            _draft("adversarial-1", perspective="failure-paths"),
            _draft("dynamic-1", perspective="authentication"),
            _draft("invented-9", perspective="extra-reviewer"),
        )
    )

    plan = compile_review_plan(
        snapshot_id=SNAPSHOT_ID,
        risk_level=RiskLevel.HIGH,
        proposal=proposal,
        allowed_files=("src/api.py",),
        allowed_symbols=(),
        allowed_hunks=(),
    )

    assert [assignment.role_kind for assignment in plan.assignments] == [
        ReviewerRoleKind.CORE,
        ReviewerRoleKind.ADVERSARIAL,
        ReviewerRoleKind.DYNAMIC,
    ]
    assert len(plan.assignments) == 3
    assert all(
        assignment.permissions == READ_ONLY_REVIEW_PERMISSIONS
        for assignment in plan.assignments
    )
    assert all(assignment.snapshot_id == SNAPSHOT_ID for assignment in plan.assignments)
    assert "invented" not in plan.to_json()


def test_critical_dynamic_assignments_always_have_distinct_perspectives() -> None:
    proposal = AssignmentPlannerProposal(
        assignments=(
            _draft("dynamic-1", perspective="security"),
            _draft("dynamic-2", perspective="security"),
        )
    )

    plan = compile_review_plan(
        snapshot_id=SNAPSHOT_ID,
        risk_level=RiskLevel.CRITICAL,
        proposal=proposal,
        allowed_files=("src/api.py",),
        allowed_symbols=(),
        allowed_hunks=(),
    )
    perspectives = [
        assignment.perspective
        for assignment in plan.assignments
        if assignment.role_kind is ReviewerRoleKind.DYNAMIC
    ]

    assert len(perspectives) == 2
    assert None not in perspectives
    assert len(set(perspectives)) == 2


def test_unauthorized_planner_targets_fall_back_to_snapshot_authorized_scope() -> None:
    proposal = AssignmentPlannerProposal(
        assignments=(
            _draft("core-1", files=("secret/outside.py",)),
        )
    )

    plan = compile_review_plan(
        snapshot_id=SNAPSHOT_ID,
        risk_level=RiskLevel.LOW,
        proposal=proposal,
        allowed_files=("src/api.py",),
        allowed_symbols=("src/api.py::handle_request",),
        allowed_hunks=("src/api.py#hunk-0",),
    )

    assert plan.assignments[0].targets == AssignmentTargets(
        files=("src/api.py",),
        symbols=("src/api.py::handle_request",),
        hunks=("src/api.py#hunk-0",),
    )


def test_assignment_and_plan_protocols_have_no_legacy_budget_or_contract_fields() -> None:
    plan = compile_review_plan(
        snapshot_id=SNAPSHOT_ID,
        risk_level=RiskLevel.LOW,
        allowed_files=("src/api.py",),
        allowed_symbols=(),
        allowed_hunks=(),
    )

    assert [field.name for field in fields(ReviewerAssignment)] == [
        "assignment_id",
        "snapshot_id",
        "role",
        "role_kind",
        "perspective",
        "mission",
        "targets",
        "checks",
        "permissions",
    ]
    assert [field.name for field in fields(ReviewPlan)] == [
        "snapshot_id",
        "risk_level",
        "assignments",
    ]
    serialized = plan.to_dict()
    forbidden = {
        "risk_reasons",
        "signal_refs",
        "contract",
        "max_turns",
        "max_tool_calls",
        "max_output_tokens",
        "max_total_tokens",
        "provider",
        "model",
    }
    assert forbidden.isdisjoint(serialized)
    assert all(forbidden.isdisjoint(item) for item in serialized["assignments"])


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
        store.resolve_pr(identity, "local", "planning-task"),
        PRMetadata(title="Planning task"),
    )
    snapshot = store.create_or_load_snapshot(workspace, BASE_SHA, HEAD_SHA)
    return store, snapshot


def _diff_index(snapshot_id: str) -> DiffArtifactIndex:
    return DiffArtifactIndex(
        snapshot_id=snapshot_id,
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        patch_artifact_id="A-" + "d" * 64,
        diff_sha256="e" * 64,
        diff_size_bytes=10,
        files=(
            DiffFileIndex(
                file_index=0,
                path="src/api.py",
                previous_path=None,
                status="modify",
                additions=1,
                deletions=0,
                binary=False,
                submodule=False,
                byte_start=0,
                byte_end=10,
                hunks=(
                    DiffHunkIndex(
                        hunk_index=0,
                        old_start=1,
                        old_count=1,
                        new_start=1,
                        new_count=1,
                        byte_start=1,
                        byte_end=9,
                    ),
                ),
            ),
        ),
    )


def _changed_symbols(snapshot_id: str) -> ChangedSymbolsV2:
    return ChangedSymbolsV2(
        snapshot_id=snapshot_id,
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        analyzer="test",
        analyzer_version="1",
        analysis_configuration="{}",
        cache_key="f" * 64,
        language_coverage=(),
        symbols=(
            ChangedSymbolV2(
                path="src/api.py",
                qualified_name="handle_request",
                kind="function",
                change_type="modified",
                line_start=1,
                line_end=5,
                analyzer="test",
                analyzer_version="1",
                analysis_configuration="{}",
                language_coverage="supported",
            ),
        ),
    )


def test_review_planning_runtime_persists_plan_and_snapshot_bound_assignments(
    tmp_path: Path,
) -> None:
    store, snapshot = _workspace(tmp_path)
    risk = compile_risk_record(
        snapshot_id=snapshot.snapshot_id,
        changed_file_count=1,
        intent=IntentPacket(
            goal="Preserve request behavior.",
            source=IntentSource.EXPLICIT,
            uncertainties=(),
        ),
        model_decision=None,
    )

    plan = ReviewPlanningRuntime(store).plan(
        snapshot,
        risk,
        _diff_index(snapshot.snapshot_id),
        _changed_symbols(snapshot.snapshot_id),
    )

    assert plan.risk_level is RiskLevel.LOW
    assert len(plan.assignments) == 1
    assert (snapshot.path / "ReviewPlan" / "plan.json").is_file()
    assert (snapshot.path / "ReviewPlan" / "Assignments" / "core-1.json").is_file()
