from dataclasses import replace

from review_agent.git_repo import ChangeSummary
from review_agent.intent import build_intent_packet
from review_agent.models import ReviewProfile, ReviewRequest, RiskAssessment, RiskLevel
from review_agent.portfolio import build_portfolio_packet
from review_agent.repository_intelligence import (
    ChangedSymbol,
    RepositoryIntelligenceSnapshot,
)
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


def test_risk_packet_carries_changed_symbols_and_stable_signal_catalog():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
    )
    summary = ChangeSummary(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
        changed_files=["auth/session.py"],
        diff_stat="1 file changed, 10 insertions",
        diff_excerpt=["+def validate_session(token):"],
    )
    intent = build_intent_packet(request, summary)
    intelligence = RepositoryIntelligenceSnapshot(
        base_revision="main",
        revision="HEAD",
        changed_symbols=[
            ChangedSymbol(
                path="auth/session.py",
                qualified_name="validate_session",
                kind="function",
                change_type="added",
                line_start=1,
                line_end=2,
            )
        ],
    )

    first = build_risk_packet(
        summary,
        intent,
        {"python_compile": "passed", "mypy": "unavailable"},
        repository_intelligence=intelligence,
    )
    second = build_risk_packet(
        summary,
        intent,
        {"mypy": "unavailable", "python_compile": "passed"},
        repository_intelligence=intelligence,
    )

    assert first.changed_symbols == [
        {
            "path": "auth/session.py",
            "qualified_name": "validate_session",
            "kind": "function",
            "change_type": "added",
            "line_start": 1,
            "line_end": 2,
        }
    ]
    assert first.signal_catalog == second.signal_catalog
    assert list(first.signal_catalog) == sorted(first.signal_catalog)
    assert "changed_file:auth/session.py" in first.signal_catalog
    assert "quality_gate:mypy" in first.signal_catalog
    assert (
        "changed_symbol:auth/session.py:validate_session:added"
        in first.signal_catalog
    )


def test_risk_packet_normalizes_uncertainty_boundary_whitespace():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
    )
    summary = ChangeSummary(
        "C:/repo",
        "main",
        "HEAD",
        ["app.py"],
        "",
        ["+def changed():"],
    )
    intent = replace(
        build_intent_packet(request, summary),
        uncertainties=["provider invocation failed: StopIteration: "],
    )

    packet = build_risk_packet(summary, intent, {"python_compile": "passed"})
    assessment = LocalRiskAssessor().assess(packet)

    assert packet.intent_uncertainties == [
        "provider invocation failed: StopIteration:"
    ]
    assert packet.signal_catalog["intent_uncertainty:000"] == (
        "provider invocation failed: StopIteration:"
    )
    assert assessment.uncertainties == packet.intent_uncertainties


def test_signal_catalog_canonicalizes_diff_values_without_changing_ref_keys():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
    )
    summary = ChangeSummary(
        "C:/repo",
        "main",
        "HEAD",
        ["app.py"],
        "",
        [" def add(left, right):", "   ", "\t+return left + right"],
    )
    intent = build_intent_packet(request, summary)

    packet = build_risk_packet(summary, intent, {"python_compile": "passed"})
    assessment = LocalRiskAssessor().assess(packet)
    portfolio_packet = build_portfolio_packet(
        assessment,
        ref_allowlist=packet.signal_catalog,
        ref_catalog=packet.signal_catalog,
    )

    assert packet.diff_excerpt == [
        " def add(left, right):",
        "   ",
        "\t+return left + right",
    ]
    assert packet.signal_catalog["diff_excerpt:000"] == "def add(left, right):"
    assert "diff_excerpt:001" not in packet.signal_catalog
    assert packet.signal_catalog["diff_excerpt:002"] == "+return left + right"
    assert portfolio_packet.ref_catalog["diff_excerpt:000"] == (
        "def add(left, right):"
    )


def test_failed_quality_gate_produces_signal_ref():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD")
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["app.py"], "", ["+def changed():"])
    intent = build_intent_packet(request, summary)
    packet = build_risk_packet(summary, intent, {"python_compile": "failed"})

    assessment = LocalRiskAssessor().assess(packet)

    assert assessment.level is RiskLevel.HIGH
    assert "quality_gate:python_compile" in assessment.signal_refs
    assert assessment.uncertainties == intent.uncertainties


def test_unavailable_quality_gate_lowers_verification_strength_signal():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
    )
    summary = ChangeSummary(
        "C:/repo",
        "main",
        "HEAD",
        ["app.py"],
        "",
        ["+def changed():"],
    )
    intent = build_intent_packet(request, summary)
    packet = build_risk_packet(
        summary,
        intent,
        {"quality_gate_discovery": "error"},
    )

    assessment = LocalRiskAssessor().assess(packet)

    assert assessment.level is RiskLevel.MEDIUM
    assert "quality_gate:quality_gate_discovery" in assessment.signal_refs
    assert "verification gap" in assessment.suggested_focus


def test_unavailable_gate_does_not_lower_sensitive_path_risk():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
    )
    summary = ChangeSummary(
        "C:/repo",
        "main",
        "HEAD",
        ["auth/session.py"],
        "",
        ["+def changed():"],
    )
    intent = build_intent_packet(request, summary)
    packet = build_risk_packet(summary, intent, {"mypy": "unavailable"})

    assessment = LocalRiskAssessor().assess(packet)

    assert assessment.level is RiskLevel.HIGH
    assert "changed_file:auth/session.py" in assessment.signal_refs
    assert "quality_gate:mypy" in assessment.signal_refs
    assert "verification gap" in assessment.suggested_focus


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
        dimensions={
            "impact": "derived from changed paths",
            "blast_radius": "local",
            "reversibility": "local",
            "uncertainty": "local",
            "verification_strength": "local",
        },
        reasons=["public behavior may change"],
        signal_refs=["diff:src/app.py"],
        uncertainties=["project constraints are not explicitly declared"],
        suggested_focus=["test adequacy"],
    )

    assignments = build_assignments(assessment)

    assert len(assignments) == 2
    assert assignments[0].initial_context.observation_refs == []
    assert assignments[0].initial_context.signal_refs == ["diff:src/app.py"]

    profile = ReviewProfile.for_risk(RiskLevel.MEDIUM)
    for assignment in assignments:
        assert assignment.max_output_tokens == profile.max_output_tokens
        assert assignment.max_total_tokens == profile.max_total_tokens
        assert assignment.max_elapsed_seconds == profile.max_elapsed_seconds
        assert assignment.max_provider_attempts == profile.max_provider_attempts


def test_assignments_receive_initial_context_not_raw_evidence():
    assessment = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={
            "impact": "local",
            "blast_radius": "local",
            "reversibility": "local",
            "uncertainty": "local",
            "verification_strength": "local",
        },
        reasons=["small or documentation-only non-sensitive change set"],
        signal_refs=["diff:README.md"],
        uncertainties=["acceptance criteria are not explicitly declared"],
        suggested_focus=["intent alignment"],
    )

    assignment = build_assignments(assessment)[0]

    assert assignment.initial_context.observation_refs == []
    assert assignment.initial_context.signal_refs == ["diff:README.md"]
    assert assignment.initial_context.quality_gate_summary == {}
    assert not hasattr(assignment, "provided_evidence_refs")
    assert not hasattr(assignment, "code_ranges")
