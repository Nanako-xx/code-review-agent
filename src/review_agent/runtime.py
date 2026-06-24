from __future__ import annotations

from review_agent.models import Assignment, RiskAssessment, ReviewProfile


def build_assignments(risk_assessment: RiskAssessment) -> list[Assignment]:
    profile = ReviewProfile.for_risk(risk_assessment.level)
    roles = _role_names(profile.reviewer_roles)
    assignments: list[Assignment] = []

    for role in roles:
        assignments.append(
            Assignment(
                role=role,
                mission=_mission_for_role(role),
                assignment_reason=list(risk_assessment.reasons),
                assigned_contract=_contract_for_role(role),
                required_checks=[
                    "map changed behavior to intent",
                    "inspect direct evidence for assigned contract items",
                    "record unavailable evidence as uncertainty",
                ],
                provided_evidence_refs=list(risk_assessment.evidence_refs),
                code_ranges=[],
                max_turns=profile.max_turns_per_reviewer,
                max_tool_calls=profile.max_tool_calls_per_reviewer,
            )
        )

    return assignments


def _role_names(role_keys: list[str]) -> list[str]:
    names = {
        "core": "Core Reviewer",
        "adversarial": "Adversarial Reviewer",
        "dynamic_specialist": "Dynamic Specialist Reviewer",
        "security_specialist": "Security Specialist Reviewer",
        "domain_specialist": "Domain Specialist Reviewer",
    }
    return [names[key] for key in role_keys]


def _mission_for_role(role: str) -> str:
    missions = {
        "Core Reviewer": "Check intent alignment, behavior correctness, regression safety, and tests.",
        "Adversarial Reviewer": "Look for edge cases, bad assumptions, and production failure modes.",
        "Dynamic Specialist Reviewer": "Investigate the highest-risk focus areas selected by runtime.",
        "Security Specialist Reviewer": "Investigate authorization, authentication, data exposure, and abuse paths.",
        "Domain Specialist Reviewer": "Investigate domain invariants and operational safety.",
    }
    return missions[role]


def _contract_for_role(role: str) -> list[str]:
    if role == "Core Reviewer":
        return ["intent_alignment", "behavioral_correctness", "regression_safety", "test_adequacy"]
    if role == "Adversarial Reviewer":
        return ["behavioral_correctness", "regression_safety", "unresolved_uncertainties"]
    return ["regression_safety", "test_adequacy", "unresolved_uncertainties"]
