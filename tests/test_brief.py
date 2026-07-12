from review_agent.brief import build_review_brief, review_brief_to_dict
from review_agent.models import (
    IntentPacket,
    IntentSource,
    IntentStatus,
    QualityGateResult,
    RiskAssessment,
    RiskLevel,
)
from review_agent.reporting import render_review_brief_markdown


def test_review_brief_to_dict_contains_spec_sections_and_recommendation() -> None:
    intent = IntentPacket(
        goal="Add auth token check",
        acceptance_criteria=["reject bad token"],
        scope=["auth.py"],
        constraints=["read-only review"],
        sources={"goal": IntentSource.EXPLICIT},
        status=IntentStatus.PARTIAL,
        uncertainties=["acceptance criteria inferred from code"],
    )
    risk = RiskAssessment(
        level=RiskLevel.HIGH,
        dimensions={"impact": "auth path"},
        reasons=["auth.py changed"],
        signal_refs=["changed_file:auth.py"],
        uncertainties=["missing integration tests"],
        suggested_focus=["regression safety"],
    )
    reconciliation_payload = {
        "canonical_findings": [
            {
                "claim": "Bad token path is not covered",
                "severity": "high",
                "confidence": "medium",
                "evidence_refs": ["O-1"],
                "reviewer_indices": [0],
                "roles": ["core"],
                "suggested_action": "Add a negative-token test",
            }
        ],
        "rejected_findings": [
            {
                "reviewer_index": 1,
                "role": "adversarial",
                "claim": "Session storage changed",
                "reason": "unsupported_claim",
                "evidence_refs": [],
                "missing_evidence_refs": [],
            }
        ],
        "remaining_disagreements": ["core and adversarial disagree on token expiry"],
        "contract_coverage": [
            {
                "reviewer_index": 0,
                "role": "core",
                "contract": "Behavioral Correctness",
                "status": "covered",
                "summary": "Token happy path reviewed",
                "evidence_refs": ["O-1"],
                "unsupported_evidence_refs": [],
            }
        ],
        "evidence_quality": "verified",
    }
    completion_summary = {
        "status": "completed_with_uncertainties",
        "recommendation": "manual_review",
        "blockers": [],
        "uncertainties": ["Intent Packet partial"],
        "missing_perspectives": [],
    }
    final_risk = {
        "status": "reassessed",
        "initial_level": "high",
        "level": "critical",
        "reasons": ["verified critical finding: data loss"],
        "escalations": ["verified critical finding: data loss"],
        "deescalations": [],
        "uncertainties": [],
        "signal_refs": ["finding:data-loss"],
    }

    brief = build_review_brief(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        intent_packet=intent,
        risk_assessment=risk,
        changed_files=["auth.py"],
        quality_results=[
            QualityGateResult(
                name="python_compile",
                status="passed",
                command=["python", "-m", "compileall"],
                summary="Compiled 1 Python file",
            )
        ],
        observation_summaries={"O-1": "auth.py changed between base and head"},
        repository_intelligence_summary="Repository Intelligence\n- modified function check auth.py:1-2",
        reconciliation_payload=reconciliation_payload,
        completion_summary=completion_summary,
        final_risk_assessment=final_risk,
        incremental_priority={
            "from_revision": "b" * 40,
            "to_revision": "c" * 40,
            "changed_files": ["auth.py"],
            "diff_stat": "1 file changed",
            "diff_excerpt": ["+reject bad token"],
        },
    )

    payload = review_brief_to_dict(brief)

    assert payload["review_id"] == "review-1"
    assert payload["change_intent"]["goal"] == "Add auth token check"
    assert payload["intent_assessment"]["status"] == "partial"
    assert payload["initial_and_final_risk_assessment"]["initial"]["level"] == "high"
    assert payload["initial_and_final_risk_assessment"]["final"]["status"] == "reassessed"
    assert payload["initial_and_final_risk_assessment"]["final"]["level"] == "critical"
    assert payload["quality_gates"][0]["name"] == "python_compile"
    assert payload["change_map_and_repository_impact"]["changed_files"] == ["auth.py"]
    assert (
        payload["change_map_and_repository_impact"]["incremental_priority"][
            "changed_files"
        ]
        == ["auth.py"]
    )
    assert payload["verified_findings"][0]["claim"] == "Bad token path is not covered"
    assert payload["rejected_hypotheses"][0]["claim"] == "Session storage changed"
    assert payload["reviewer_disagreements"] == ["core and adversarial disagree on token expiry"]
    assert payload["review_contract_coverage"][0]["contract"] == "Behavioral Correctness"
    assert payload["non_binding_recommendation"] == "manual_review"
    assert "auth.py" in payload["human_review_checklist_and_reading_order"][0]
    markdown = render_review_brief_markdown(brief)
    assert "Incremental priority map:" in markdown
    assert f"{'b' * 40}..{'c' * 40}" in markdown


def test_render_review_brief_markdown_uses_spec_section_order() -> None:
    intent = IntentPacket(goal="Add auth token check", status=IntentStatus.SUFFICIENT)
    risk = RiskAssessment(
        level=RiskLevel.MEDIUM,
        dimensions={"impact": "auth path"},
        reasons=["auth.py changed"],
        signal_refs=["changed_file:auth.py"],
        uncertainties=[],
        suggested_focus=["test adequacy"],
    )
    brief = build_review_brief(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        intent_packet=intent,
        risk_assessment=risk,
        changed_files=["auth.py"],
        quality_results=[],
        completion_summary={"recommendation": "manual_review"},
    )

    markdown = render_review_brief_markdown(brief)

    expected_sections = [
        "## Change Intent",
        "## Intent Assessment",
        "## Initial And Final Risk Assessment",
        "## Quality Gates",
        "## Change Map And Repository Impact",
        "## Verified Findings",
        "## Rejected Hypotheses",
        "## Uncertainties",
        "## Reviewer Disagreements",
        "## Review Contract Coverage",
        "## Verification Evidence",
        "## Human Review Checklist And Reading Order",
        "## Non-Binding Recommendation",
    ]
    positions = [markdown.index(section) for section in expected_sections]
    assert positions == sorted(positions)
    assert "Risk level: medium" in markdown
    assert "Manual review required before merge." in markdown
