from pathlib import Path
import json

from review_agent.checkpoint import CheckpointStore
from review_agent.models import IntentStatus, ReviewerResult, ReviewerResultStatus, RiskAssessment, RiskLevel
from review_agent.reporting import render_markdown_report
from review_agent.run_state import initial_run_state


def test_checkpoint_store_writes_json_and_jsonl(tmp_path: Path):
    store = CheckpointStore(tmp_path, review_id="review-1")

    store.write_json("request.json", {"base": "main", "head": "HEAD"})
    store.append_jsonl("observations.jsonl", {"observation_id": "O-1", "summary": "changed app.py"})

    assert json.loads((tmp_path / ".review-agent" / "runs" / "review-1" / "request.json").read_text(encoding="utf-8")) == {
        "base": "main",
        "head": "HEAD",
    }
    assert "O-1" in (tmp_path / ".review-agent" / "runs" / "review-1" / "observations.jsonl").read_text(
        encoding="utf-8"
    )


def test_checkpoint_store_serializes_enum_values(tmp_path: Path):
    store = CheckpointStore(tmp_path, review_id="review-1")

    store.write_json("intent.json", {"status": IntentStatus.PARTIAL})

    assert json.loads((tmp_path / ".review-agent" / "runs" / "review-1" / "intent.json").read_text(encoding="utf-8")) == {
        "status": "partial",
    }


def test_checkpoint_store_writes_and_reads_run_state(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "review-1")
    state = initial_run_state(
        review_id="review-1",
        repository_path=str(tmp_path),
        base_revision="main",
        head_revision="HEAD",
    )

    store.write_state(state)

    assert store.read_state() == state


def test_markdown_report_contains_risk_signals_and_uncertainties():
    assessment = RiskAssessment(
        level=RiskLevel.HIGH,
        dimensions={"impact": "sensitive path"},
        reasons=["sensitive path changed: auth.py"],
        signal_refs=["changed_file:auth.py"],
        uncertainties=["user did not provide explicit intent"],
        suggested_focus=["regression safety"],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
    )

    assert "# Review Brief" in report
    assert "Risk level: high" in report
    assert "## Initial And Final Risk Assessment" in report
    assert "- changed_file:auth.py" in report
    assert "## Uncertainties" in report
    assert "- user did not provide explicit intent" in report


def test_markdown_report_includes_single_reviewer_result():
    assessment = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={"impact": "local"},
        reasons=["small change"],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=["intent alignment"],
    )
    reviewer_result = ReviewerResult(
        investigation_summary="Fake reviewer executed.",
        status=ReviewerResultStatus.PARTIAL,
        uncertainties=["Fake provider does not perform semantic review."],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
        reviewer_result=reviewer_result,
    )

    assert "## Uncertainties" in report
    assert "- Fake provider does not perform semantic review." in report


def test_markdown_report_includes_observation_summaries():
    assessment = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={"impact": "local"},
        reasons=[],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=[],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
        observation_summaries={"O-abc": "auth.py changed between base and head"},
    )

    assert "## Verification Evidence" in report
    assert "- observation:O-abc - auth.py changed between base and head" in report


def test_markdown_report_includes_repository_intelligence_summary():
    assessment = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={},
        reasons=[],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=[],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["app.py"],
        repository_intelligence_summary="Repository Intelligence\n- modified function add app.py:1-2",
    )

    assert "## Change Map And Repository Impact" in report
    assert "Repository intelligence:" in report
    assert "modified function add app.py:1-2" in report


def test_markdown_report_includes_multi_reviewer_summary():
    assessment = RiskAssessment(
        level=RiskLevel.MEDIUM,
        dimensions={},
        reasons=[],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=[],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
        multi_reviewer_summary={
            "reviewer_count": 2,
            "status_counts": {"partial": 2},
            "roles": ["Core Reviewer", "Adversarial Reviewer"],
        },
    )

    assert "## Change Map And Repository Impact" in report
    assert "reviewer_count: 2" in report
    assert "roles: Core Reviewer, Adversarial Reviewer" in report


def test_markdown_report_includes_reconciliation_and_completion_sections():
    assessment = RiskAssessment(
        level=RiskLevel.HIGH,
        dimensions={},
        reasons=[],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=[],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
        reconciliation_summary={
            "canonical_count": 1,
            "rejected_count": 0,
            "evidence_quality": "verified",
        },
        completion_summary={
            "status": "completed",
            "recommendation": "needs_work",
            "blockers": [],
            "uncertainties": [],
            "missing_perspectives": [],
        },
    )

    assert "## Verified Findings" in report
    assert "## Rejected Hypotheses" in report
    assert "## Non-Binding Recommendation" in report
    assert "Needs work before merge." in report
