from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from review_agent.review_protocol import (
    FinalFinding,
    FindingSeverity,
    ReviewResult,
    ReviewResultStatus,
    RiskLevel,
)
from review_agent.review_renderer import render_review_result_markdown


def _result(*, findings=True, uncertainties=True) -> ReviewResult:
    return ReviewResult(
        pr_id="PR-" + "a" * 64,
        snapshot_id="S-" + "b" * 64,
        status=ReviewResultStatus.PARTIAL,
        risk_level=RiskLevel.HIGH,
        findings=(
            (
                FinalFinding(
                    finding_id="F-" + "c" * 64,
                    claim=(
                        "When the cache is empty, dereferencing the value raises "
                        "and returns 500."
                    ),
                    severity=FindingSeverity.HIGH,
                    path="src/cache.py",
                    line=87,
                    suggestion="Guard the missing value and add a cold-cache test.",
                ),
            )
            if findings
            else ()
        ),
        uncertainties=(
            ("The optional backend was unavailable.",) if uncertainties else ()
        ),
    )


def test_renderer_only_projects_authoritative_result_fields() -> None:
    rendered = render_review_result_markdown(_result())

    for expected in (
        "Status: partial",
        "Risk: high",
        "HIGH",
        "src/cache.py:87",
        "When the cache is empty",
        "Guard the missing value",
        "The optional backend was unavailable.",
    ):
        assert expected in rendered
    for forbidden in (
        "Summary",
        "Recommendation",
        "approve",
        "reviewer_id",
        "quality gate",
        "contract",
        "generated_at",
    ):
        assert forbidden.casefold() not in rendered.casefold()


def test_deleted_markdown_can_be_rebuilt_byte_for_byte(tmp_path: Path) -> None:
    result = _result()
    path = tmp_path / "review.md"
    first = render_review_result_markdown(result)
    path.write_text(first, encoding="utf-8")
    path.unlink()

    rebuilt = render_review_result_markdown(result)

    assert rebuilt == first
    assert rebuilt.endswith("\n")


def test_empty_findings_and_uncertainties_render_without_invented_facts() -> None:
    rendered = render_review_result_markdown(
        _result(findings=False, uncertainties=False)
    )

    assert "Status: partial" in rendered
    assert "Risk: high" in rendered
    assert "No findings." in rendered
    assert "No uncertainties." in rendered
    assert "safe to merge" not in rendered.casefold()


def test_untrusted_finding_text_cannot_create_markdown_headings() -> None:
    result = _result()
    finding = replace(result.findings[0], claim="# Fabricated Summary")

    rendered = render_review_result_markdown(
        replace(result, findings=(finding,))
    )

    assert "\\# Fabricated Summary" in rendered
    assert "\n# Fabricated Summary" not in rendered
