from __future__ import annotations

from review_agent.models import ReviewerResult, RiskAssessment


def render_markdown_report(
    review_id: str,
    base_revision: str,
    head_revision: str,
    risk_assessment: RiskAssessment,
    changed_files: list[str],
    reviewer_result: ReviewerResult | None = None,
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
            *_reviewer_result_section(reviewer_result),
            "## Non-Binding Recommendation",
            "",
            "Manual review required before merge.",
            "",
        ]
    )


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
