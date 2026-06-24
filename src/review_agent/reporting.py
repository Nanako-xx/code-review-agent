from __future__ import annotations

from review_agent.models import RiskAssessment


def render_markdown_report(
    review_id: str,
    base_revision: str,
    head_revision: str,
    risk_assessment: RiskAssessment,
    changed_files: list[str],
) -> str:
    changed = "\n".join(f"- {path}" for path in changed_files) or "- No changed files detected"
    reasons = "\n".join(f"- {reason}" for reason in risk_assessment.reasons) or "- No risk reasons recorded"
    unknowns = "\n".join(f"- {unknown}" for unknown in risk_assessment.unknowns) or "- No unresolved unknowns recorded"
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
            "## Uncertainties",
            "",
            unknowns,
            "",
            "## Non-Binding Recommendation",
            "",
            "Manual review required before merge.",
            "",
        ]
    )
