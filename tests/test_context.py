from review_agent.context import build_reviewer_envelope
from review_agent.models import Assignment, InitialContext, IntentPacket, IntentSource, IntentStatus


def test_reviewer_envelope_uses_standard_four_inputs():
    assignment = Assignment(
        role="Core Reviewer",
        mission="Check intent alignment",
        assignment_reason=["small non-sensitive change set"],
        assigned_contract=["intent_alignment"],
        required_checks=["map changed behavior to intent"],
        initial_context=InitialContext(
            changed_files=["app.py"],
            diff_ranges=["app.py:1-5"],
            code_ranges=["app.py:1-5"],
            quality_gate_summary={"python_compile": "passed"},
            observation_refs=["O-diff-app"],
        ),
        max_turns=6,
        max_tool_calls=12,
    )
    intent = IntentPacket(
        goal="Review changes touching app.py",
        sources={"goal": IntentSource.INFERRED},
        status=IntentStatus.PARTIAL,
        uncertainties=["user did not provide user intent"],
    )

    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={"app.py:1-5": "def add(a, b):\n    return a + b\n"},
        observations={"O-diff-app": "app.py changed between base and head"},
        trace_id="trace-1",
    )

    assert set(envelope.__dict__.keys()) == {"system", "tools", "messages", "parameters"}
    assert "tools" not in envelope.messages[0]
    assert envelope.parameters["trace_id"] == "trace-1"
    assert "Review Contract" in envelope.system
    assert "risk_level" not in str(envelope.messages)
    assert "Assignment" in envelope.messages[0]["content"]
    assert "Observation Summary" in envelope.messages[0]["content"]
    assert "Initial Context" in envelope.messages[0]["content"]
    assert "Evidence" not in envelope.messages[0]["content"]
    assert "explicit" not in envelope.messages[0]["content"]
    assert "inferred" in envelope.messages[0]["content"]


def test_reviewer_tools_describe_head_default_and_base_head_comparison():
    assignment = Assignment(
        role="Core Reviewer",
        mission="Check intent alignment",
        assignment_reason=["small non-sensitive change set"],
        assigned_contract=["intent_alignment"],
        required_checks=["map changed behavior to intent"],
        initial_context=InitialContext(),
        max_turns=6,
        max_tool_calls=12,
    )
    intent = IntentPacket(goal="Review changes", status=IntentStatus.PARTIAL)

    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={},
        observations={},
        trace_id="trace-2",
    )

    tool_text = " ".join(str(tool) for tool in envelope.tools)
    assert "head revision" in tool_text
    assert "base and head" in tool_text


def test_reviewer_envelope_includes_repository_intelligence_tools():
    assignment = Assignment(
        role="Core Reviewer",
        mission="Check intent alignment",
        assignment_reason=[],
        assigned_contract=[],
        required_checks=[],
        initial_context=InitialContext(),
        max_turns=6,
        max_tool_calls=12,
    )
    intent = IntentPacket(goal="Review changes", status=IntentStatus.PARTIAL)

    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={},
        observations={},
        trace_id="trace-ri-tools",
    )

    tool_names = {tool["name"] for tool in envelope.tools}
    assert {"list_symbols", "inspect_symbol", "find_references"}.issubset(tool_names)
