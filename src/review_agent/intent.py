from __future__ import annotations

from review_agent.git_repo import ChangeSummary
from review_agent.models import IntentPacket, IntentSource, IntentStatus, ReviewRequest


def build_intent_packet(request: ReviewRequest, change_summary: ChangeSummary) -> IntentPacket:
    unknowns: list[str] = []
    sources: dict[str, IntentSource] = {}

    if request.user_intent:
        goal = request.user_intent
        sources["goal"] = IntentSource.DECLARED
    elif change_summary.changed_files:
        files = ", ".join(change_summary.changed_files[:3])
        goal = f"Review changes touching {files}"
        sources["goal"] = IntentSource.INFERRED
        unknowns.append("user did not provide declared intent")
    else:
        goal = None
        unknowns.append("no changed files were detected")

    acceptance_criteria: list[str] = []
    unknowns.append("acceptance criteria are not explicitly declared")

    scope = list(change_summary.changed_files)
    if scope:
        sources["scope"] = IntentSource.INFERRED

    constraints: list[str] = []
    if request.project_rules:
        constraints.extend(request.project_rules)
        sources["constraints"] = IntentSource.DECLARED
    else:
        unknowns.append("project constraints are not explicitly declared")

    if goal is None:
        status = IntentStatus.INSUFFICIENT
    elif acceptance_criteria and constraints:
        status = IntentStatus.SUFFICIENT
    else:
        status = IntentStatus.PARTIAL

    return IntentPacket(
        goal=goal,
        acceptance_criteria=acceptance_criteria,
        scope=scope,
        constraints=constraints,
        sources=sources,
        status=status,
        unknowns=unknowns,
    )
