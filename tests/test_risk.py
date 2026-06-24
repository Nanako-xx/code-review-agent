from review_agent.git_repo import ChangeSummary
from review_agent.intent import build_intent_packet
from review_agent.models import ReviewRequest, RiskLevel
from review_agent.risk import LocalRiskAssessor, build_risk_packet


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
