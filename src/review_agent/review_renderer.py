from __future__ import annotations

import re

from review_agent.review_protocol import ReviewResult


_MARKDOWN_CONTROL = re.compile(r"([\\`*_\[\]<>#])")


def render_review_result_markdown(result: ReviewResult) -> str:
    """Render only facts already present in the authoritative ReviewResult."""

    if type(result) is not ReviewResult:
        raise ValueError("result must be ReviewResult")
    lines = [
        "# Code Review",
        "",
        f"Status: {result.status.value}",
        f"Risk: {result.risk_level.value}",
        "",
        "## Findings",
        "",
    ]
    if not result.findings:
        lines.append("No findings.")
    else:
        for index, finding in enumerate(result.findings, start=1):
            if index > 1:
                lines.append("")
            lines.extend(
                (
                    f"### {finding.severity.value.upper()} — "
                    f"`{_escape_markdown(finding.path)}:{finding.line}`",
                    "",
                    _escape_markdown(finding.claim),
                    "",
                    f"Suggestion: {_escape_markdown(finding.suggestion)}",
                )
            )
    lines.extend(("", "## Uncertainties", ""))
    if result.uncertainties:
        lines.extend(
            f"- {_escape_markdown(value)}" for value in result.uncertainties
        )
    else:
        lines.append("No uncertainties.")
    return "\n".join(lines) + "\n"


def _escape_markdown(value: str) -> str:
    return _MARKDOWN_CONTROL.sub(r"\\\1", value)


__all__ = ["render_review_result_markdown"]
