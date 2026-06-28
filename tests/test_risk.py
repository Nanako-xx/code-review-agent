from review_agent.git_repo import ChangeSummary
from review_agent.intent import build_intent_packet
from review_agent.models import ReviewRequest, RiskAssessment, RiskLevel
from review_agent.risk import LocalRiskAssessor, build_risk_packet
from review_agent.runtime import build_assignments


def test_risk_packet_carries_intent_uncertainties():
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
    assert packet.intent_status == intent.status
    assert packet.intent_uncertainties == intent.uncertainties
    assert packet.diff_excerpt == ["+def validate_session(token):", "+    return token is not None"]


def test_failed_quality_gate_produces_signal_ref():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD")
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["app.py"], "", ["+def changed():"])
    intent = build_intent_packet(request, summary)
    packet = build_risk_packet(summary, intent, {"python_compile": "failed"})

    assessment = LocalRiskAssessor().assess(packet)

    assert assessment.level is RiskLevel.HIGH
    assert "quality_gate:python_compile" in assessment.signal_refs
    assert assessment.uncertainties == intent.uncertainties


def test_many_doc_files_do_not_become_medium_risk_by_count_only():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD")
    changed_files = [f"docs/note-{index}.md" for index in range(10)]
    summary = ChangeSummary("C:/repo", "main", "HEAD", changed_files, "10 files changed", ["+docs"])
    intent = build_intent_packet(request, summary)
    packet = build_risk_packet(summary, intent, {"python_compile": "passed"})

    assessment = LocalRiskAssessor().assess(packet)

    assert assessment.level is RiskLevel.LOW
    assert "many files changed" not in " ".join(assessment.reasons)


def test_sensitive_path_still_high_risk():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD")
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["auth/session.py"], "", ["+def validate_session(token):"])
    intent = build_intent_packet(request, summary)
    packet = build_risk_packet(summary, intent, {"python_compile": "passed"})

    assessment = LocalRiskAssessor().assess(packet)

    assert assessment.level is RiskLevel.HIGH
    assert "sensitive path changed: auth/session.py" in assessment.reasons
    assert "changed_file:auth/session.py" in assessment.signal_refs
    assert "caller compatibility" in assessment.suggested_focus


def test_runtime_assignments_use_initial_context():
    assessment = RiskAssessment(
        level=RiskLevel.MEDIUM,
        dimensions={"impact": "derived from changed paths"},
        reasons=["public behavior may change"],
        signal_refs=["diff:src/app.py"],
        uncertainties=["project constraints are not explicitly declared"],
        suggested_focus=["test adequacy"],
    )

    assignments = build_assignments(assessment)

    assert len(assignments) == 2
    assert assignments[0].initial_context.observation_refs == ["diff:src/app.py"]


def test_assignments_receive_initial_context_not_raw_evidence():
    assessment = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={"impact": "local"},
        reasons=["small or documentation-only non-sensitive change set"],
        signal_refs=["diff:README.md"],
        uncertainties=["acceptance criteria are not explicitly declared"],
        suggested_focus=["intent alignment"],
    )

    assignment = build_assignments(assessment)[0]

    assert assignment.initial_context.observation_refs == ["diff:README.md"]
    assert assignment.initial_context.quality_gate_summary == {}
    assert not hasattr(assignment, "provided_evidence_refs")
    assert not hasattr(assignment, "code_ranges")
