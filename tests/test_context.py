import pytest

from review_agent.context import (
    ContextBudget,
    build_reviewer_context_payload,
    build_reviewer_envelope,
)
from review_agent.models import (
    Assignment,
    ClarificationQuestion,
    ClarificationStatus,
    InitialContext,
    IntentClaim,
    IntentConfidence,
    IntentField,
    IntentOrigin,
    IntentPacket,
    IntentSource,
    IntentStatus,
)


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
            signal_refs=["changed_file:app.py"],
        ),
        max_turns=6,
        max_tool_calls=12,
        max_output_tokens=1234,
        max_total_tokens=9876,
        max_elapsed_seconds=90,
        max_provider_attempts=3,
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
    assert "4096 output tokens per model call" in envelope.messages[0]["content"]
    assert "65536 total tokens" in envelope.messages[0]["content"]
    assert "300 elapsed seconds" in envelope.messages[0]["content"]
    assert "2 provider attempts per model turn" in envelope.messages[0]["content"]
    assert "Evidence" not in envelope.messages[0]["content"]
    assert "explicit" not in envelope.messages[0]["content"]
    assert "inferred" in envelope.messages[0]["content"]


def test_reviewer_context_injects_complete_intent_with_stable_compact_metadata():
    goal_claim = IntentClaim(
        field=IntentField.GOAL,
        value="Prevent duplicate retry jobs",
        source=IntentSource.EXPLICIT,
        origin=IntentOrigin.USER_INPUT,
        confidence=IntentConfidence.HIGH,
        source_refs=["request:user_intent"],
    )
    acceptance_claim = IntentClaim(
        field=IntentField.ACCEPTANCE_CRITERIA,
        value="Reject duplicate job IDs",
        source=IntentSource.INFERRED,
        origin=IntentOrigin.REPOSITORY_TEST,
        confidence=IntentConfidence.MEDIUM,
        evidence_refs=["O-retry-tests"],
    )
    scope_claim = IntentClaim(
        field=IntentField.SCOPE,
        value="src/retry.py",
        source=IntentSource.INFERRED,
        origin=IntentOrigin.LLM_INFERENCE,
        confidence=IntentConfidence.LOW,
        evidence_refs=["O-retry-diff"],
    )
    constraint_claim = IntentClaim(
        field=IntentField.CONSTRAINTS,
        value="Keep storage schema compatible",
        source=IntentSource.EXPLICIT,
        origin=IntentOrigin.PROJECT_RULE,
        confidence=IntentConfidence.HIGH,
        source_refs=["AGENTS.md"],
    )
    open_question = ClarificationQuestion(
        field=IntentField.SCOPE,
        question="Should worker changes be included?",
        rationale="Worker behavior can change retry correctness.",
        proposed_values=["src/worker.py"],
        claim_ids=[scope_claim.claim_id],
        status=ClarificationStatus.OPEN,
    )
    resolved_question = ClarificationQuestion(
        field=IntentField.CONSTRAINTS,
        question="Must the storage schema remain compatible?",
        rationale="A schema change would alter the compatibility conclusion.",
        proposed_values=["Keep storage schema compatible"],
        claim_ids=[constraint_claim.claim_id],
        status=ClarificationStatus.CONFIRMED,
        resolved_values=["Keep storage schema compatible"],
        decision_id="decision-storage-compatibility",
    )
    intent = IntentPacket(
        goal="Prevent duplicate retry jobs",
        acceptance_criteria=["Reject duplicate job IDs", "Preserve successful retries"],
        scope=["src/retry.py", "tests/test_retry.py"],
        constraints=["Keep storage schema compatible"],
        sources={
            "constraints": IntentSource.EXPLICIT,
            "scope": IntentSource.INFERRED,
            "acceptance_criteria": IntentSource.INFERRED,
            "goal": IntentSource.EXPLICIT,
        },
        status=IntentStatus.PARTIAL,
        uncertainties=["Worker scope remains unconfirmed"],
        provenance=[constraint_claim, scope_claim, acceptance_claim, goal_claim],
        clarifications=[resolved_question, open_question],
    )

    result = build_reviewer_context_payload(
        assignment=_context_assignment(),
        intent=intent,
        code_snippets={},
        observations={},
    )

    assert "Risk Signal Refs: changed_file:app.py" in result.messages[0]["content"]
    intent_section = result.messages[0]["content"].split("\n\n")[1]
    assert intent_section == "\n".join(
        [
            "Intent Packet",
            "Goal: Prevent duplicate retry jobs",
            "Acceptance Criteria: Reject duplicate job IDs; Preserve successful retries",
            "Scope: src/retry.py; tests/test_retry.py",
            "Constraints: Keep storage schema compatible",
            "Status: partial",
            "Sources: goal=explicit, acceptance_criteria=inferred, scope=inferred, constraints=explicit",
            "Claim Provenance:",
            "- goal | explicit/user_input | high | active | material | source=request:user_intent; evidence=none | Prevent duplicate retry jobs",
            "- acceptance_criteria | inferred/repository_test | medium | active | material | source=none; evidence=O-retry-tests | Reject duplicate job IDs",
            "- scope | inferred/llm_inference | low | active | material | source=none; evidence=O-retry-diff | src/retry.py",
            "- constraints | explicit/project_rule | high | active | material | source=AGENTS.md; evidence=none | Keep storage schema compatible",
            "Open Clarifications:",
            "- scope | open | proposed=src/worker.py | question=Should worker changes be included? | rationale=Worker behavior can change retry correctness.",
            "Uncertainties: Worker scope remains unconfirmed",
        ]
    )


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


def test_reviewer_envelope_exposes_only_runtime_allowed_tools():
    envelope = build_reviewer_envelope(
        assignment=_context_assignment(),
        intent=_context_intent(),
        code_snippets={},
        observations={},
        trace_id="trace-limited-tools",
        allowed_tools=("read_range", "search_code"),
    )

    assert [tool["name"] for tool in envelope.tools] == [
        "search_code",
        "read_range",
    ]
    assert envelope.parameters["tool_choice"] == "auto"


def test_reviewer_envelope_can_disable_tools_and_rejects_unknown_allowlist_items():
    envelope = build_reviewer_envelope(
        assignment=_context_assignment(),
        intent=_context_intent(),
        code_snippets={},
        observations={},
        trace_id="trace-no-tools",
        allowed_tools=(),
    )

    assert envelope.tools == []
    assert envelope.parameters["tool_choice"] == "none"

    with pytest.raises(ValueError, match="unsupported reviewer tool"):
        build_reviewer_envelope(
            assignment=_context_assignment(),
            intent=_context_intent(),
            code_snippets={},
            observations={},
            trace_id="trace-invalid-tools",
            allowed_tools=("write_file",),
        )


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
    assert result.metadata["compressed_sections"]
    assert set(result.metadata["compressed_sections"]).issubset(
        {"Code Snippets", "Observation Summary"}
    )


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


def test_context_budget_rejects_invalid_limits():
    with pytest.raises(ValueError, match="max_message_chars"):
        ContextBudget(max_message_chars=0)
    with pytest.raises(ValueError, match="compacted_section_min_chars"):
        ContextBudget(compacted_section_min_chars=0)


def test_context_payload_metadata_marks_whole_payload_compaction():
    result = build_reviewer_context_payload(
        assignment=_context_assignment(),
        intent=_context_intent(),
        code_snippets={"app.py:1-20": "x = 1\n" * 200},
        observations={"O-1": "observation\n" * 200},
        context_budget=ContextBudget(max_message_chars=300, compacted_section_min_chars=80),
    )

    content = result.messages[0]["content"]

    assert result.metadata["message_chars"] <= 300
    assert "Context Payload" in result.metadata["compressed_sections"]
    assert all(section in content for section in result.metadata["included_sections"])
    assert result.metadata["whole_payload_compacted"] is True
    normal_compressed = set(result.metadata["compressed_sections"]) - {"Context Payload"}
    assert normal_compressed.isdisjoint(result.metadata["omitted_sections"])


def test_whole_payload_compaction_does_not_infer_sections_from_body_text():
    assignment = _context_assignment()
    assignment = Assignment(
        role=assignment.role,
        mission="Mentions Completion Rules before the real section",
        assignment_reason=assignment.assignment_reason,
        assigned_contract=assignment.assigned_contract,
        required_checks=assignment.required_checks,
        initial_context=assignment.initial_context,
        max_turns=assignment.max_turns,
        max_tool_calls=assignment.max_tool_calls,
    )

    result = build_reviewer_context_payload(
        assignment=assignment,
        intent=_context_intent(),
        code_snippets={"app.py:1-20": "x = 1\n" * 200},
        observations={"O-1": "observation\n" * 200},
        context_budget=ContextBudget(max_message_chars=300, compacted_section_min_chars=80),
    )

    assert result.metadata["whole_payload_compacted"] is True
    assert "Completion Rules" not in result.metadata["included_sections"]


def test_whole_payload_compaction_requires_full_section_header_for_inclusion():
    result = build_reviewer_context_payload(
        assignment=_context_assignment(),
        intent=_context_intent(),
        code_snippets={"app.py:1-20": "x = 1\n" * 200},
        observations={"O-1": "observation\n" * 200},
        context_budget=ContextBudget(max_message_chars=83, compacted_section_min_chars=80),
    )

    assert result.metadata["whole_payload_compacted"] is True
    assert result.metadata["included_sections"] == []
    assert "Assignment" in result.metadata["omitted_sections"]
    assert result.metadata["message_chars"] <= result.metadata["max_message_chars"]


def test_reviewer_envelope_records_context_metadata():
    envelope = build_reviewer_envelope(
        assignment=_context_assignment(),
        intent=_context_intent(),
        code_snippets={"app.py:1-20": "x = 1\n" * 200},
        observations={"O-1": "app.py changed\n" * 200},
        trace_id="trace-context-metadata",
        context_budget=ContextBudget(max_message_chars=1500),
    )

    metadata = envelope.parameters["context"]

    assert metadata["budget_scope"] == "messages_only"
    assert metadata["message_chars"] <= 1500
    assert "system" in metadata["excluded_from_budget"]
    assert "tools" in metadata["excluded_from_budget"]
    assert "parameters" in metadata["excluded_from_budget"]
    assert envelope.messages[0]["role"] == "user"
    assert len(envelope.messages[0]["content"]) == metadata["message_chars"]


def test_reviewer_envelope_uses_explicit_model_parameters():
    envelope = build_reviewer_envelope(
        assignment=_context_assignment(),
        intent=_context_intent(),
        code_snippets={},
        observations={},
        trace_id="trace-model-params",
        model="deepseek-chat",
        max_output_tokens=2048,
        reasoning_effort="high",
    )

    assert envelope.parameters["model"] == "deepseek-chat"
    assert envelope.parameters["max_output_tokens"] == 2048
    assert envelope.parameters["max_elapsed_seconds"] == 90
    assert envelope.parameters["reasoning_effort"] == "high"
    assert envelope.parameters["response_schema"] == "reviewer_assignment_result_v2"


def test_reviewer_envelope_defaults_model_limits_from_assignment():
    assignment = _context_assignment()

    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=_context_intent(),
        code_snippets={},
        observations={},
        trace_id="trace-assignment-budget",
    )

    assert envelope.parameters["max_output_tokens"] == assignment.max_output_tokens
    assert envelope.parameters["max_elapsed_seconds"] == assignment.max_elapsed_seconds
