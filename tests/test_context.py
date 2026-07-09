from review_agent.context import ContextBudget, build_reviewer_context_payload, build_reviewer_envelope
from review_agent.models import Assignment, InitialContext, IntentPacket, IntentSource, IntentStatus


def _context_assignment() -> Assignment:
    return Assignment(
        role="Core Reviewer",
        mission="Check intent alignment",
        assignment_reason=["runtime expanded low risk into one core reviewer"],
        assigned_contract=["intent_alignment"],
        required_checks=["map changed behavior to intent"],
        initial_context=InitialContext(
            changed_files=["app.py"],
            diff_ranges=["app.py:1-200"],
            code_ranges=["app.py:1-200"],
            quality_gate_summary={"python_compile": "passed"},
            observation_refs=["O-diff-app"],
        ),
        max_turns=6,
        max_tool_calls=12,
    )


def _context_intent() -> IntentPacket:
    return IntentPacket(
        goal="Review changes touching app.py",
        sources={"goal": IntentSource.INFERRED},
        status=IntentStatus.PARTIAL,
        uncertainties=["user did not provide user intent"],
    )


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


def test_context_payload_compacts_variable_sections_to_message_budget():
    huge_snippet = "\n".join(f"line {index}: return value_{index}" for index in range(300))
    huge_observation = "\n".join(f"observation {index}" for index in range(300)) + "\ntail-marker"

    result = build_reviewer_context_payload(
        assignment=_context_assignment(),
        intent=_context_intent(),
        code_snippets={"app.py:1-200": huge_snippet},
        observations={"O-huge": huge_observation},
        context_budget=ContextBudget(max_message_chars=1800),
    )

    content = result.messages[0]["content"]

    assert result.metadata["max_message_chars"] == 1800
    assert result.metadata["message_chars"] <= 1800
    assert "Assignment" in content
    assert "Intent Packet" in content
    assert "Completion Rules" in content
    assert "[compacted" in content
    assert "tail-marker" not in content
    assert "Code Snippets" in result.metadata["compressed_sections"]
    assert "Observation Summary" in result.metadata["compressed_sections"]


def test_context_budget_applies_to_messages_only_not_tools_or_parameters():
    result = build_reviewer_context_payload(
        assignment=_context_assignment(),
        intent=_context_intent(),
        code_snippets={"app.py:1-20": "x = 1\n" * 200},
        observations={},
        context_budget=ContextBudget(max_message_chars=1200),
    )

    assert result.metadata["message_chars"] <= 1200
    assert result.metadata["budget_scope"] == "messages_only"
    assert result.metadata["excluded_from_budget"] == ["system", "tools", "parameters"]
