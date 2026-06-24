from review_agent.git_repo import ChangeSummary
from review_agent.intent import build_intent_packet
from review_agent.models import ReviewRequest, RiskAssessment, RiskLevel
from review_agent.risk import LocalRiskAssessor, build_risk_packet
from review_agent.runtime import build_assignments


def test_build_risk_packet_is_lightweight():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD")
    summary = ChangeSummary(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
        changed_files=["auth/session.py"],
        diff_stat="1 file changed, 10 insertions",
        diff_excerpt=["+def validate_session(token):", "+    return token is not None"],
    )
    intent = build_intent_packet(request, summary)

    packet = build_risk_packet(summary, intent, quality_gate_status={"python_compile": "passed"})

    assert packet.change_summary["changed_files"] == ["auth/session.py"]
    assert packet.deterministic_signals["quality_gates"] == {"python_compile": "passed"}
    assert packet.diff_excerpt == ["+def validate_session(token):", "+    return token is not None"]


def test_local_risk_assessor_marks_sensitive_paths_high():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD")
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["auth/session.py"], "", ["+def validate_session(token):"])
    intent = build_intent_packet(request, summary)
    packet = build_risk_packet(summary, intent, quality_gate_status={})

    assessment = LocalRiskAssessor().assess(packet)

    assert assessment.level is RiskLevel.HIGH
    assert "sensitive path changed: auth/session.py" in assessment.reasons
    assert "caller compatibility" in assessment.suggested_focus


def test_local_risk_assessor_marks_small_non_sensitive_changes_low():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD", user_intent="Update docs")
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["README.md"], "1 file changed", ["+new docs"])
    intent = build_intent_packet(request, summary)
    packet = build_risk_packet(summary, intent, quality_gate_status={"python_compile": "not_applicable"})

    assessment = LocalRiskAssessor().assess(packet)

    assert assessment.level is RiskLevel.LOW
    assert assessment.suggested_focus == ["intent alignment", "changed file sanity"]


def test_runtime_expands_high_risk_into_specific_assignments():
    assessment = RiskAssessment(
        level=RiskLevel.HIGH,
        dimensions={},
        reasons=["sensitive path changed: auth/session.py"],
        evidence_refs=[],
        unknowns=["project constraints are not explicitly declared"],
        suggested_focus=["caller compatibility", "regression safety", "test adequacy"],
    )

    assignments = build_assignments(assessment)

    assert len(assignments) == 3
    assert assignments[0].role == "Core Reviewer"
    assert assignments[1].role == "Adversarial Reviewer"
    assert assignments[2].role == "Dynamic Specialist Reviewer"
    assert assignments[0].max_turns == 16
    assert assignments[0].required_checks == [
        "map changed behavior to intent",
        "inspect direct evidence for assigned contract items",
        "record unavailable evidence as uncertainty",
    ]
