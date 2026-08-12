from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

from review_agent.diff_artifact import DiffArtifactIndex
from review_agent.portfolio import (
    AssignmentPlannerDraft,
    AssignmentPlannerProposal,
)
from review_agent.pr_workspace import PRWorkspaceStore, SnapshotWorkspace
from review_agent.repository_intelligence import ChangedSymbolsV2
from review_agent.review_protocol import (
    AssignmentTargets,
    ReviewPlan,
    ReviewerAssignment,
    ReviewerRoleKind,
    RiskLevel,
)
from review_agent.risk_runtime import RiskRecord
from review_agent.safe_io import canonical_json_bytes


READ_ONLY_REVIEW_PERMISSIONS = (
    "read_range",
    "compare_base_head",
    "search_code",
    "list_symbols",
    "inspect_symbol",
    "find_references",
    "read_commit_messages",
    "read_artifact",
)


@dataclass(frozen=True)
class ReviewerSlot:
    slot_id: str
    role: str
    role_kind: ReviewerRoleKind


_CORE_SLOT = ReviewerSlot("core-1", "Core Reviewer", ReviewerRoleKind.CORE)
_ADVERSARIAL_SLOT = ReviewerSlot(
    "adversarial-1",
    "Adversarial Reviewer",
    ReviewerRoleKind.ADVERSARIAL,
)
_DYNAMIC_SLOT_1 = ReviewerSlot(
    "dynamic-1", "Dynamic Reviewer", ReviewerRoleKind.DYNAMIC
)
_DYNAMIC_SLOT_2 = ReviewerSlot(
    "dynamic-2", "Dynamic Reviewer", ReviewerRoleKind.DYNAMIC
)
_SLOTS_BY_RISK = {
    RiskLevel.LOW: (_CORE_SLOT,),
    RiskLevel.MEDIUM: (_CORE_SLOT, _ADVERSARIAL_SLOT),
    RiskLevel.HIGH: (_CORE_SLOT, _ADVERSARIAL_SLOT, _DYNAMIC_SLOT_1),
    RiskLevel.CRITICAL: (
        _CORE_SLOT,
        _ADVERSARIAL_SLOT,
        _DYNAMIC_SLOT_1,
        _DYNAMIC_SLOT_2,
    ),
}

_DEFAULT_MISSIONS = {
    ReviewerRoleKind.CORE: (
        "Review Intent alignment, primary business correctness, caller "
        "compatibility, regressions, and the main tests."
    ),
    ReviewerRoleKind.ADVERSARIAL: (
        "Challenge error paths, boundaries, concurrency, retries, idempotency, "
        "partial failure, resource cleanup, and recovery."
    ),
    ReviewerRoleKind.DYNAMIC: (
        "Review the assigned specialist perspective without duplicating the "
        "complete Core review."
    ),
}
_DEFAULT_CHECKS = {
    ReviewerRoleKind.CORE: (
        "Verify the implementation matches the Intent.",
        "Verify caller compatibility and primary regression coverage.",
    ),
    ReviewerRoleKind.ADVERSARIAL: (
        "Verify boundary and failure-path behavior.",
        "Verify retry, cleanup, and partial-failure behavior.",
    ),
    ReviewerRoleKind.DYNAMIC: (
        "Verify the assigned specialist invariants.",
    ),
}
_DEFAULT_DYNAMIC_PERSPECTIVES = {
    "dynamic-1": "highest-risk-domain",
    "dynamic-2": "cross-component-invariants",
}


class ReviewPlanningError(ValueError):
    pass


class ReviewPlanningIntegrityError(ReviewPlanningError):
    pass


def fixed_reviewer_slots(level: RiskLevel) -> tuple[ReviewerSlot, ...]:
    if not isinstance(level, RiskLevel):
        raise ReviewPlanningError("risk level must be a RiskLevel")
    return _SLOTS_BY_RISK[level]


def _authorized_targets(
    *,
    files: Iterable[str],
    symbols: Iterable[str],
    hunks: Iterable[str],
) -> AssignmentTargets:
    return AssignmentTargets(
        files=tuple(dict.fromkeys(files)),
        symbols=tuple(dict.fromkeys(symbols)),
        hunks=tuple(dict.fromkeys(hunks)),
    )


def _draft_is_authorized(
    draft: AssignmentPlannerDraft,
    authorized: AssignmentTargets,
) -> bool:
    targets = draft.targets
    if not (targets.files or targets.symbols or targets.hunks):
        return not (authorized.files or authorized.symbols or authorized.hunks)
    return (
        set(targets.files).issubset(authorized.files)
        and set(targets.symbols).issubset(authorized.symbols)
        and set(targets.hunks).issubset(authorized.hunks)
    )


def _assignment_id(
    slot: ReviewerSlot,
    *,
    snapshot_id: str,
    perspective: str | None,
    mission: str,
    targets: AssignmentTargets,
    checks: tuple[str, ...],
) -> str:
    identity = {
        "snapshot_id": snapshot_id,
        "slot_id": slot.slot_id,
        "role": slot.role,
        "role_kind": slot.role_kind.value,
        "perspective": perspective,
        "mission": mission,
        "targets": targets.to_dict(),
        "checks": list(checks),
        "permissions": list(READ_ONLY_REVIEW_PERMISSIONS),
    }
    return "ASG-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def compile_review_plan(
    *,
    snapshot_id: str,
    risk_level: RiskLevel,
    proposal: AssignmentPlannerProposal | None = None,
    allowed_files: Iterable[str],
    allowed_symbols: Iterable[str],
    allowed_hunks: Iterable[str],
) -> ReviewPlan:
    slots = fixed_reviewer_slots(risk_level)
    if proposal is not None and type(proposal) is not AssignmentPlannerProposal:
        raise ReviewPlanningError(
            "proposal must be an AssignmentPlannerProposal or null"
        )
    authorized = _authorized_targets(
        files=allowed_files,
        symbols=allowed_symbols,
        hunks=allowed_hunks,
    )
    drafts = {
        draft.slot_id: draft
        for draft in (() if proposal is None else proposal.assignments)
        if draft.slot_id in {slot.slot_id for slot in slots}
    }

    assignments: list[ReviewerAssignment] = []
    dynamic_perspectives: set[str] = set()
    for slot in slots:
        candidate = drafts.get(slot.slot_id)
        draft = (
            candidate
            if candidate is not None and _draft_is_authorized(candidate, authorized)
            else None
        )
        mission = (
            draft.mission if draft is not None else _DEFAULT_MISSIONS[slot.role_kind]
        )
        targets = draft.targets if draft is not None else authorized
        checks = (
            draft.checks
            if draft is not None and draft.checks
            else _DEFAULT_CHECKS[slot.role_kind]
        )
        perspective = draft.perspective if draft is not None else None
        if slot.role_kind is ReviewerRoleKind.DYNAMIC:
            if perspective is None or perspective in dynamic_perspectives:
                perspective = _DEFAULT_DYNAMIC_PERSPECTIVES[slot.slot_id]
            if perspective in dynamic_perspectives:
                perspective = f"{perspective}-{slot.slot_id}"
            dynamic_perspectives.add(perspective)

        assignments.append(
            ReviewerAssignment(
                assignment_id=_assignment_id(
                    slot,
                    snapshot_id=snapshot_id,
                    perspective=perspective,
                    mission=mission,
                    targets=targets,
                    checks=checks,
                ),
                snapshot_id=snapshot_id,
                role=slot.role,
                role_kind=slot.role_kind,
                perspective=perspective,
                mission=mission,
                targets=targets,
                checks=checks,
                permissions=READ_ONLY_REVIEW_PERMISSIONS,
            )
        )

    return ReviewPlan(
        snapshot_id=snapshot_id,
        risk_level=risk_level,
        assignments=tuple(assignments),
    )


class ReviewPlanningRuntime:
    def __init__(self, workspace_store: PRWorkspaceStore) -> None:
        if not isinstance(workspace_store, PRWorkspaceStore):
            raise ReviewPlanningError(
                "Review Planning Runtime requires a PRWorkspaceStore"
            )
        self._store = workspace_store

    def plan(
        self,
        snapshot: SnapshotWorkspace,
        risk: RiskRecord,
        diff_index: DiffArtifactIndex,
        changed_symbols: ChangedSymbolsV2,
        *,
        proposal: AssignmentPlannerProposal | None = None,
    ) -> ReviewPlan:
        self._store.verify_snapshot(snapshot)
        if not isinstance(risk, RiskRecord):
            raise ReviewPlanningError("risk must be a RiskRecord")
        if risk.snapshot_id != snapshot.snapshot_id:
            raise ReviewPlanningIntegrityError(
                "Risk Snapshot binding does not match Review Plan"
            )
        if not isinstance(diff_index, DiffArtifactIndex):
            raise ReviewPlanningError("diff_index must be a DiffArtifactIndex")
        if diff_index.snapshot_id != snapshot.snapshot_id:
            raise ReviewPlanningIntegrityError(
                "DiffArtifact Snapshot binding does not match Review Plan"
            )
        if not isinstance(changed_symbols, ChangedSymbolsV2):
            raise ReviewPlanningError(
                "changed_symbols must be ChangedSymbolsV2"
            )
        if changed_symbols.snapshot_id != snapshot.snapshot_id:
            raise ReviewPlanningIntegrityError(
                "ChangedSymbols Snapshot binding does not match Review Plan"
            )

        allowed_files = tuple(file.path for file in diff_index.files)
        allowed_symbols = tuple(
            f"{symbol.path}::{symbol.qualified_name}"
            for symbol in changed_symbols.symbols
        )
        allowed_hunks = tuple(
            f"{file.path}#hunk-{hunk.hunk_index}"
            for file in diff_index.files
            for hunk in file.hunks
        )
        plan = compile_review_plan(
            snapshot_id=snapshot.snapshot_id,
            risk_level=risk.final_level,
            proposal=proposal,
            allowed_files=allowed_files,
            allowed_symbols=allowed_symbols,
            allowed_hunks=allowed_hunks,
        )
        slots = fixed_reviewer_slots(risk.final_level)
        for slot, assignment in zip(slots, plan.assignments):
            self._store.publish_create_only(
                snapshot,
                f"ReviewPlan/Assignments/{slot.slot_id}.json",
                assignment.to_json_bytes(),
            )
        self._store.publish_create_only(
            snapshot,
            "ReviewPlan/plan.json",
            plan.to_json_bytes(),
        )
        return plan


__all__ = [
    "READ_ONLY_REVIEW_PERMISSIONS",
    "ReviewPlanningError",
    "ReviewPlanningIntegrityError",
    "ReviewPlanningRuntime",
    "ReviewerSlot",
    "compile_review_plan",
    "fixed_reviewer_slots",
]
