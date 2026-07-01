from __future__ import annotations

from review_agent.models import ReviewerResult, RiskAssessment


def render_markdown_report(
    review_id: str,
    base_revision: str,
    head_revision: str,
    risk_assessment: RiskAssessment,
    changed_files: list[str],
    reviewer_result: ReviewerResult | None = None,
    observation_summaries: dict[str, str] | None = None,
    repository_intelligence_summary: str | None = None,
    multi_reviewer_summary: dict[str, object] | None = None,
    reconciliation_summary: dict[str, object] | None = None,
    completion_summary: dict[str, object] | None = None,
) -> str:
    changed = "\n".join(f"- {path}" for path in changed_files) or "- No changed files detected"
    reasons = "\n".join(f"- {reason}" for reason in risk_assessment.reasons) or "- No risk reasons recorded"
    signals = "\n".join(f"- {ref}" for ref in risk_assessment.signal_refs) or "- No risk signals recorded"
    uncertainties = (
        "\n".join(f"- {uncertainty}" for uncertainty in risk_assessment.uncertainties)
        or "- No unresolved uncertainties recorded"
    )
    focus = "\n".join(f"- {item}" for item in risk_assessment.suggested_focus) or "- No suggested focus recorded"

    return "\n".join(
        [
            "# Review Brief",
            "",
            f"Review ID: {review_id}",
            f"Base: {base_revision}",
            f"Head: {head_revision}",
            f"Risk level: {risk_assessment.level.value}",
            "",
            "## Changed Files",
            "",
            changed,
            "",
            "## Risk Reasons",
            "",
            reasons,
            "",
            "## Suggested Review Focus",
            "",
            focus,
            "",
            "## Risk Signals",
            "",
            signals,
            "",
            "## Uncertainties",
            "",
            uncertainties,
            "",
            *_observation_section(observation_summaries or {}),
            *_repository_intelligence_section(repository_intelligence_summary),
            *_multi_reviewer_section(multi_reviewer_summary),
            *_reconciliation_section(reconciliation_summary),
            *_completion_section(completion_summary),
            *_reviewer_result_section(reviewer_result),
            "## Non-Binding Recommendation",
            "",
            "Manual review required before merge.",
            "",
        ]
    )


def _reconciliation_section(reconciliation_summary: dict[str, object] | None) -> list[str]:
    if not reconciliation_summary:
        return []
    return [
        "## Evidence Reconciliation",
        "",
        f"Canonical findings: {reconciliation_summary.get('canonical_count', 0)}",
        f"Rejected findings: {reconciliation_summary.get('rejected_count', 0)}",
        f"Evidence quality: {reconciliation_summary.get('evidence_quality', 'unknown')}",
        "",
    ]


def _completion_section(completion_summary: dict[str, object] | None) -> list[str]:
    if not completion_summary:
        return []
    blockers = _markdown_list(completion_summary.get("blockers", []), "No blockers recorded")
    uncertainties = _markdown_list(completion_summary.get("uncertainties", []), "No completion uncertainties recorded")
    missing = _markdown_list(
        completion_summary.get("missing_perspectives", []),
        "No missing reviewer perspectives recorded",
    )
    return [
        "## Completion Status",
        "",
        f"Status: {completion_summary.get('status', 'unknown')}",
        f"Recommendation: {completion_summary.get('recommendation', 'manual_review')}",
        "",
        "### Completion Blockers",
        "",
        blockers,
        "",
        "### Completion Uncertainties",
        "",
        uncertainties,
        "",
        "### Missing Perspectives",
        "",
        missing,
        "",
    ]


def _multi_reviewer_section(multi_reviewer_summary: dict[str, object] | None) -> list[str]:
    if not multi_reviewer_summary:
        return []
    roles = multi_reviewer_summary.get("roles", [])
    status_counts = multi_reviewer_summary.get("status_counts", {})
    role_lines = "\n".join(f"- {role}" for role in roles) or "- No reviewer roles recorded"
    status_lines = (
        "\n".join(f"- {status}: {count}" for status, count in dict(status_counts).items())
        or "- No reviewer statuses recorded"
    )
    return [
        "## Multi-Reviewer Summary",
        "",
        f"Reviewers: {multi_reviewer_summary.get('reviewer_count', 0)}",
        "",
        "### Reviewer Roles",
        "",
        role_lines,
        "",
        "### Reviewer Status Counts",
        "",
        status_lines,
        "",
    ]


def _markdown_list(items: object, fallback: str) -> str:
    if not items:
        return f"- {fallback}"
    return "\n".join(f"- {item}" for item in items)


def _repository_intelligence_section(repository_intelligence_summary: str | None) -> list[str]:
    if not repository_intelligence_summary:
        return []
    return [
        "## Repository Intelligence",
        "",
        repository_intelligence_summary,
        "",
    ]


def _observation_section(observation_summaries: dict[str, str]) -> list[str]:
    if not observation_summaries:
        return []
    items = "\n".join(
        f"- {observation_id}: {summary}" for observation_id, summary in observation_summaries.items()
    )
    return [
        "## Observations",
        "",
        items,
        "",
    ]


def _reviewer_result_section(reviewer_result: ReviewerResult | None) -> list[str]:
    if reviewer_result is None:
        return []
    findings = (
        "\n".join(f"- {finding.claim}" for finding in reviewer_result.confirmed_findings)
        or "- No confirmed findings reported by the single reviewer"
    )
    uncertainties = (
        "\n".join(f"- {uncertainty}" for uncertainty in reviewer_result.uncertainties)
        or "- No reviewer uncertainties recorded"
    )
    return [
        "## Single Reviewer Result",
        "",
        f"Status: {reviewer_result.status.value}",
        f"Summary: {reviewer_result.investigation_summary}",
        "",
        "### Reviewer Findings",
        "",
        findings,
        "",
        "### Reviewer Uncertainties",
        "",
        uncertainties,
        "",
    ]
