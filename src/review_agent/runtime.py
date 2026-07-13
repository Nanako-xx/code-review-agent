from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterable

from review_agent.models import Assignment, InitialContext, RiskAssessment, RiskLevel, ReviewProfile
from review_agent.portfolio import (
    ADVERSARIAL_REVIEW_CONTRACT,
    CORE_REVIEW_CONTRACT,
    SPECIALIST_REVIEW_CONTRACT,
    PortfolioCandidate,
    PortfolioPacket,
    PortfolioPlannerRun,
    PortfolioProposal,
    build_portfolio_packet,
    deterministic_fallback_proposal,
    validate_portfolio_proposal,
)


_REQUIRED_CHECKS = (
    "map changed behavior to intent",
    "inspect direct observations for assigned Contract items",
    "record unavailable observations as uncertainty",
)
_ROLE_REQUIRED_CHECKS = {
    "core": ("inspect affected callers or record why unavailable",),
    "adversarial": ("challenge happy-path assumptions and boundary behavior",),
    "specialist": ("trace the assigned specialist perspective through affected behavior",),
}
_ROLE_ORDER = {"core": 0, "adversarial": 1, "specialist": 2}
_SPECIALIST_MINIMUM = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 0,
    RiskLevel.HIGH: 1,
    RiskLevel.CRITICAL: 2,
}


@dataclass(frozen=True)
class PortfolioCompilation:
    assignments: list[Assignment]
    planner_status: str
    planner_source: str
    summary: str
    uncertainties: list[str]
    policy_actions: list[str]
    selected_candidate_ids: list[str]
    rejected_candidate_ids: list[str]
    proposed_candidate_count: int
    final_reviewer_count: int
    minimum_reviewers: int
    maximum_reviewers: int
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.planner_status not in {"accepted", "local", "disabled", "fallback"}:
            raise ValueError("planner_status is unsupported")
        if self.planner_source not in {"model", "local"}:
            raise ValueError("planner_source is unsupported")
        if self.planner_status == "fallback" and not self.fallback_reason:
            raise ValueError("fallback compilation must contain fallback_reason")
        if self.planner_status != "fallback" and self.fallback_reason is not None:
            raise ValueError("non-fallback compilation cannot contain fallback_reason")
        if not isinstance(self.assignments, list) or any(
            not isinstance(assignment, Assignment) for assignment in self.assignments
        ):
            raise ValueError("assignments must be a list of Assignment")
        for name, value in {
            "proposed_candidate_count": self.proposed_candidate_count,
            "final_reviewer_count": self.final_reviewer_count,
            "minimum_reviewers": self.minimum_reviewers,
            "maximum_reviewers": self.maximum_reviewers,
        }.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.final_reviewer_count != len(self.assignments):
            raise ValueError("final_reviewer_count must match assignments")
        if not self.minimum_reviewers <= self.final_reviewer_count <= self.maximum_reviewers:
            raise ValueError("final reviewer count violates Runtime bounds")
        _require_string_list(self.uncertainties, "uncertainties")
        _require_string_list(self.policy_actions, "policy_actions")
        _require_string_list(self.selected_candidate_ids, "selected_candidate_ids")
        _require_string_list(self.rejected_candidate_ids, "rejected_candidate_ids")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("summary must be a non-empty string")
        object.__setattr__(self, "assignments", list(self.assignments))
        object.__setattr__(self, "uncertainties", list(self.uncertainties))
        object.__setattr__(self, "policy_actions", list(self.policy_actions))
        object.__setattr__(self, "selected_candidate_ids", list(self.selected_candidate_ids))
        object.__setattr__(self, "rejected_candidate_ids", list(self.rejected_candidate_ids))

    @property
    def status(self) -> str:
        return self.planner_status

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignments": [asdict(assignment) for assignment in self.assignments],
            "planner_status": self.planner_status,
            "planner_source": self.planner_source,
            "summary": self.summary,
            "uncertainties": list(self.uncertainties),
            "policy_actions": list(self.policy_actions),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "proposed_candidate_count": self.proposed_candidate_count,
            "final_reviewer_count": self.final_reviewer_count,
            "minimum_reviewers": self.minimum_reviewers,
            "maximum_reviewers": self.maximum_reviewers,
            "fallback_reason": self.fallback_reason,
        }


# Naming alias for callers that treat the compiler output as the portfolio plan artifact.
PortfolioPlan = PortfolioCompilation


@dataclass(frozen=True)
class _CandidateSelection:
    candidate: PortfolioCandidate
    source: str


class PortfolioCompiler:
    """Small facade around the deterministic compiler for dependency injection."""

    def compile(
        self,
        packet: PortfolioPacket,
        proposal: PortfolioProposal | None = None,
        *,
        planner_run: PortfolioPlannerRun | None = None,
        planner_source: str | None = None,
    ) -> PortfolioCompilation:
        return compile_portfolio(
            packet,
            proposal,
            planner_run=planner_run,
            planner_source=planner_source,
        )


def compile_portfolio(
    packet: PortfolioPacket,
    proposal: PortfolioProposal | None = None,
    *,
    planner_run: PortfolioPlannerRun | None = None,
    planner_source: str | None = None,
) -> PortfolioCompilation:
    """Compile an untrusted proposal into the authoritative Assignment portfolio."""

    if not isinstance(packet, PortfolioPacket):
        raise ValueError("packet must be a PortfolioPacket")
    if proposal is not None and planner_run is not None:
        raise ValueError("proposal and planner_run are mutually exclusive")
    if planner_source is not None and planner_source not in {"model", "local"}:
        raise ValueError("planner_source must be model or local")

    policy_actions = [
        (
            "enforced_reviewer_bounds:"
            f"{packet.risk_level.value}:{packet.minimum_reviewers}-{packet.maximum_reviewers}"
        ),
        f"enforced_risk_budget:{packet.risk_level.value}",
        "enforced_permissions:read_only:safe_checks_only",
        "enforced_core_contract:" + ",".join(CORE_REVIEW_CONTRACT),
    ]
    fallback_reason: str | None = None
    proposed_candidate_count = 0

    if planner_run is not None:
        if not isinstance(planner_run, PortfolioPlannerRun):
            raise ValueError("planner_run must be a PortfolioPlannerRun")
        if planner_run.status == "accepted":
            effective_proposal = planner_run.proposal
            assert effective_proposal is not None
            proposed_candidate_count = len(effective_proposal.candidates)
            effective_source = "model"
            planner_status = "accepted"
        else:
            fallback_reason = planner_run.failure_reason or "portfolio planner failed"
            effective_proposal = deterministic_fallback_proposal(packet)
            effective_source = "local"
            planner_status = "fallback"
            policy_actions.append("planner_fallback:deterministic_candidates")
    elif proposal is not None:
        effective_proposal = proposal
        if isinstance(proposal, PortfolioProposal):
            proposed_candidate_count = len(proposal.candidates)
        effective_source = planner_source or "model"
        planner_status = "local" if effective_source == "local" else "accepted"
    else:
        effective_proposal = deterministic_fallback_proposal(packet)
        effective_source = "local"
        planner_status = "disabled"
        policy_actions.append("planner_disabled:deterministic_candidates")

    try:
        validate_portfolio_proposal(effective_proposal, packet=packet)
    except ValueError as error:
        fallback_reason = f"Runtime rejected portfolio proposal: {error}"
        effective_proposal = deterministic_fallback_proposal(packet)
        effective_source = "local"
        planner_status = "fallback"
        policy_actions.append("planner_fallback:runtime_validation")

    selections, selection_actions, rejected_ids = _select_candidates(
        packet,
        effective_proposal,
        effective_source,
    )
    policy_actions.extend(selection_actions)
    assignments = [
        _build_assignment(packet, selection)
        for selection in sorted(selections, key=_selection_order)
    ]

    assignment_ids = [assignment.assignment_id for assignment in assignments]
    if len(assignment_ids) != len(set(assignment_ids)):
        raise RuntimeError("stable Assignment ID collision")

    uncertainties = list(effective_proposal.uncertainties)
    if fallback_reason is not None:
        uncertainties.append(f"Portfolio planner fallback: {fallback_reason}")

    return PortfolioCompilation(
        assignments=assignments,
        planner_status=planner_status,
        planner_source=("model" if planner_status == "accepted" else "local"),
        summary=effective_proposal.summary,
        uncertainties=_dedupe(uncertainties),
        policy_actions=_dedupe(policy_actions),
        selected_candidate_ids=[selection.candidate.candidate_id for selection in sorted(selections, key=_selection_order)],
        rejected_candidate_ids=_dedupe(rejected_ids),
        proposed_candidate_count=proposed_candidate_count,
        final_reviewer_count=len(assignments),
        minimum_reviewers=packet.minimum_reviewers,
        maximum_reviewers=packet.maximum_reviewers,
        fallback_reason=fallback_reason,
    )


def build_assignments(
    risk_assessment: RiskAssessment,
    *,
    packet: PortfolioPacket | None = None,
    proposal: PortfolioProposal | None = None,
) -> list[Assignment]:
    """Backward-compatible local entry point, now routed through the compiler."""

    if not isinstance(risk_assessment, RiskAssessment):
        raise ValueError("risk_assessment must be a RiskAssessment")
    resolved_packet = packet or build_portfolio_packet(risk_assessment)
    if resolved_packet.risk_level is not risk_assessment.level:
        raise ValueError("packet risk level must match risk_assessment")
    return compile_portfolio(
        resolved_packet,
        proposal,
        planner_source=("model" if proposal is not None else None),
    ).assignments


def portfolio_plan_to_dict(plan: PortfolioCompilation) -> dict[str, Any]:
    if not isinstance(plan, PortfolioCompilation):
        raise ValueError("plan must be a PortfolioCompilation")
    return plan.to_dict()


def stable_assignment_id(role_kind: str, perspective_key: str) -> str:
    if role_kind not in _ROLE_ORDER:
        raise ValueError("role_kind is unsupported")
    if not isinstance(perspective_key, str) or not perspective_key.strip():
        raise ValueError("perspective_key must be a non-empty string")
    canonical = _canonical_perspective(perspective_key)
    encoded = json.dumps(
        ["portfolio_assignment_v1", role_kind, canonical],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"assignment_{digest}"


def _select_candidates(
    packet: PortfolioPacket,
    proposal: PortfolioProposal,
    source: str,
) -> tuple[list[_CandidateSelection], list[str], list[str]]:
    actions: list[str] = []
    rejected: list[str] = []
    candidates = sorted(proposal.candidates, key=_candidate_preference)

    by_perspective: dict[str, PortfolioCandidate] = {}
    for candidate in candidates:
        perspective = _canonical_perspective(candidate.perspective_key)
        kept = by_perspective.get(perspective)
        if kept is not None:
            rejected.append(candidate.candidate_id)
            actions.append(
                "deduplicated_perspective:"
                f"{candidate.candidate_id}:kept={kept.candidate_id}"
            )
            continue
        by_perspective[perspective] = candidate

    unique = sorted(by_perspective.values(), key=_candidate_preference)
    fallback_candidates = deterministic_fallback_proposal(packet).candidates
    selections: list[_CandidateSelection] = []

    required_roles = ["core"]
    if packet.risk_level is not RiskLevel.LOW:
        required_roles.append("adversarial")

    for role_kind in ("core", "adversarial"):
        role_candidates = [candidate for candidate in unique if candidate.role_kind == role_kind]
        if role_kind in required_roles:
            if role_candidates:
                selections.append(_CandidateSelection(role_candidates[0], source))
                surplus = role_candidates[1:]
            else:
                injected = _fallback_candidate(
                    packet,
                    role_kind,
                    fallback_candidates,
                    selections,
                )
                selections.append(_CandidateSelection(injected, "runtime_injected"))
                actions.append(f"injected_required_role:{role_kind}:{injected.candidate_id}")
                surplus = []
        else:
            surplus = role_candidates
        for candidate in surplus:
            rejected.append(candidate.candidate_id)
            actions.append(f"rejected_surplus_role:{role_kind}:{candidate.candidate_id}")

    specialist_candidates = [
        candidate for candidate in unique if candidate.role_kind == "specialist"
    ]
    specialist_capacity = packet.maximum_reviewers - len(selections)
    required_specialists = _SPECIALIST_MINIMUM[packet.risk_level]
    selected_specialists = specialist_candidates[:specialist_capacity]

    while len(selected_specialists) < required_specialists:
        injected = _fallback_candidate(
            packet,
            "specialist",
            fallback_candidates,
            [*selections, *(_CandidateSelection(item, source) for item in selected_specialists)],
        )
        selected_specialists.append(injected)
        actions.append(
            f"injected_required_role:specialist:{injected.candidate_id}"
        )

    injected_ids = {
        candidate.candidate_id
        for candidate in selected_specialists
        if candidate not in specialist_candidates
    }
    selections.extend(
        _CandidateSelection(
            candidate,
            "runtime_injected" if candidate.candidate_id in injected_ids else source,
        )
        for candidate in selected_specialists
    )

    selected_specialist_ids = {candidate.candidate_id for candidate in selected_specialists}
    for candidate in specialist_candidates:
        if candidate.candidate_id in selected_specialist_ids:
            continue
        rejected.append(candidate.candidate_id)
        actions.append(f"rejected_maximum_slots:{candidate.candidate_id}")

    if not packet.minimum_reviewers <= len(selections) <= packet.maximum_reviewers:
        raise RuntimeError("Runtime candidate selection violated reviewer count bounds")
    return selections, actions, rejected


def _fallback_candidate(
    packet: PortfolioPacket,
    role_kind: str,
    fallback_candidates: list[PortfolioCandidate],
    selections: list[_CandidateSelection],
) -> PortfolioCandidate:
    used_perspectives = {
        _canonical_perspective(selection.candidate.perspective_key)
        for selection in selections
    }
    for candidate in fallback_candidates:
        if candidate.role_kind != role_kind:
            continue
        if _canonical_perspective(candidate.perspective_key) not in used_perspectives:
            return candidate
    if role_kind != "specialist":
        raise RuntimeError(f"Runtime fallback does not contain required role: {role_kind}")

    index = 1
    while f"runtime-specialist-{index}" in used_perspectives:
        index += 1
    return PortfolioCandidate(
        candidate_id=f"runtime-generic-specialist-{index}",
        role_kind="specialist",
        role_name="Runtime Specialist Reviewer",
        perspective_key=f"runtime-specialist-{index}",
        mission="Investigate the highest-risk focus selected by Runtime.",
        reason_refs=list(packet.risk_signal_refs),
        context_refs=list(packet.risk_signal_refs),
        required_checks=[
            "trace the selected risk focus through affected behavior",
            "record unavailable specialist evidence as uncertainty",
        ],
        priority=60 - min(index, 50),
    )


def _build_assignment(
    packet: PortfolioPacket,
    selection: _CandidateSelection,
) -> Assignment:
    candidate = selection.candidate
    profile = ReviewProfile.for_risk(packet.risk_level)
    baseline_contract = {
        "core": CORE_REVIEW_CONTRACT,
        "adversarial": ADVERSARIAL_REVIEW_CONTRACT,
        "specialist": SPECIALIST_REVIEW_CONTRACT,
    }[candidate.role_kind]
    assigned_contract = _dedupe([*baseline_contract, *candidate.extra_contract])
    required_checks = _dedupe(
        [
            *_REQUIRED_CHECKS,
            *_ROLE_REQUIRED_CHECKS[candidate.role_kind],
            *candidate.required_checks,
        ]
    )
    assignment_reason = _dedupe([*packet.risk_reasons, *candidate.reason_refs])
    authorized_context_refs = _dedupe(
        [
            *packet.risk_signal_refs,
            *candidate.reason_refs,
            *candidate.context_refs,
        ]
    )
    observation_refs = [
        ref for ref in authorized_context_refs if _is_observation_id(ref)
    ]
    signal_refs = [
        ref for ref in authorized_context_refs if not _is_observation_id(ref)
    ]
    changed_files = _changed_files(packet.change_map)

    return Assignment(
        role=candidate.role_name,
        mission=candidate.mission,
        assignment_reason=assignment_reason,
        assigned_contract=assigned_contract,
        required_checks=required_checks,
        initial_context=InitialContext(
            changed_files=changed_files,
            observation_refs=observation_refs,
            signal_refs=signal_refs,
        ),
        max_turns=profile.max_turns_per_reviewer,
        max_tool_calls=profile.max_tool_calls_per_reviewer,
        max_output_tokens=profile.max_output_tokens,
        max_total_tokens=profile.max_total_tokens,
        max_elapsed_seconds=profile.max_elapsed_seconds,
        max_provider_attempts=profile.max_provider_attempts,
        repository_permission="read_only",
        command_permission="safe_checks_only",
        assignment_id=stable_assignment_id(
            candidate.role_kind,
            candidate.perspective_key,
        ),
        role_kind=candidate.role_kind,
        perspective_key=candidate.perspective_key,
        planner_source=selection.source,
    )


def _changed_files(change_map: dict[str, Any]) -> list[str]:
    value = change_map.get("changed_files", [])
    if not isinstance(value, list):
        return []
    return _dedupe(
        item
        for item in value
        if isinstance(item, str) and item.strip() and item == item.strip()
    )


def _candidate_preference(candidate: PortfolioCandidate) -> tuple[int, int, str, str]:
    return (
        _ROLE_ORDER[candidate.role_kind],
        -candidate.priority,
        _canonical_perspective(candidate.perspective_key),
        candidate.candidate_id,
    )


def _selection_order(selection: _CandidateSelection) -> tuple[int, int, str, str]:
    return _candidate_preference(selection.candidate)


def _canonical_perspective(value: str) -> str:
    return " ".join(value.split()).casefold()


def _is_observation_id(value: str) -> bool:
    # ObservationStore owns O-* identities. Every other authorized Portfolio ref
    # is a planning signal and must never be presented as evidence.
    return value.startswith("O-")


def _require_string_list(value: object, name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} items must be non-empty strings")


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
