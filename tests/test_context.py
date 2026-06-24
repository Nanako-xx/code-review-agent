from review_agent.context import build_reviewer_envelope
from review_agent.models import Assignment, IntentPacket, IntentSource, IntentStatus


def test_reviewer_envelope_uses_standard_four_inputs():
    assignment = Assignment(
        role="Core Reviewer",
        mission="Check intent alignment",
        assignment_reason=["small non-sensitive change set"],
        assigned_contract=["intent_alignment"],
        required_checks=["map changed behavior to intent"],
        provided_evidence_refs=["ev_1"],
        code_ranges=["app.py:1-5"],
        max_turns=6,
        max_tool_calls=12,
    )
    intent = IntentPacket(
        goal="Review changes touching app.py",
        sources={"goal": IntentSource.INFERRED},
        status=IntentStatus.PARTIAL,
        unknowns=["user did not provide declared intent"],
    )

    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={"app.py:1-5": "def add(a, b):\n    return a + b\n"},
        evidence={"ev_1": "app.py changed in head revision"},
        trace_id="trace-1",
    )

    assert set(envelope.__dict__.keys()) == {"system", "tools", "messages", "parameters"}
    assert "tools" not in envelope.messages[0]
    assert envelope.parameters["trace_id"] == "trace-1"
    assert "Review Contract" in envelope.system
    assert "risk_level" not in str(envelope.messages)
    assert "Assignment" in envelope.messages[0]["content"]
    assert "Evidence" in envelope.messages[0]["content"]
