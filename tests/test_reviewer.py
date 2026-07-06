import pytest

from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnResponse
from review_agent.models import Assignment, InitialContext, IntentPacket, IntentSource, IntentStatus, ReviewerResultStatus
from review_agent.reviewer import ReviewerResultParseError, parse_reviewer_result, run_single_reviewer


def test_parse_reviewer_result_accepts_valid_json():
    result = parse_reviewer_result(
        """
        {
          "contract_assessments": [
            {
              "contract": "intent_alignment",
              "status": "covered",
              "summary": "The change matches the stated intent.",
              "evidence_refs": ["O-diff-auth"]
            }
          ],
          "confirmed_findings": [
            {
              "claim": "The admin check now always returns true.",
              "severity": "high",
              "confidence": "high",
              "evidence_refs": ["O-diff-auth"],
              "suggested_action": "Restore the role check."
            }
          ],
          "rejected_hypotheses": ["No caller compatibility issue found in the provided context."],
          "uncertainties": ["No repository-wide caller search was available in this slice."],
          "observation_refs": ["O-diff-auth"],
          "investigation_summary": "Reviewed the assignment, intent, diff excerpt, and observations.",
          "status": "completed"
        }
        """
    )

    assert result.status is ReviewerResultStatus.COMPLETED
    assert result.confirmed_findings[0].claim == "The admin check now always returns true."
    assert result.contract_assessments[0].evidence_refs == ["O-diff-auth"]
    assert result.observation_refs == ["O-diff-auth"]


def test_parse_reviewer_result_strips_markdown_json_fence():
    result = parse_reviewer_result(
        """```json
        {
          "contract_assessments": [],
          "confirmed_findings": [],
          "rejected_hypotheses": [],
          "uncertainties": ["needs broader repository context"],
          "observation_refs": [],
          "investigation_summary": "No finding.",
          "status": "partial"
        }
        ```"""
    )

    assert result.status is ReviewerResultStatus.PARTIAL
    assert result.uncertainties == ["needs broader repository context"]


def test_parse_reviewer_result_rejects_missing_required_keys():
    with pytest.raises(ReviewerResultParseError, match="missing required key: status"):
        parse_reviewer_result(
            """
            {
              "contract_assessments": [],
              "confirmed_findings": [],
              "rejected_hypotheses": [],
              "uncertainties": [],
              "observation_refs": [],
              "investigation_summary": "No status."
            }
            """
        )


def test_parse_reviewer_result_rejects_invalid_status():
    with pytest.raises(ReviewerResultParseError, match="invalid reviewer status"):
        parse_reviewer_result(
            """
            {
              "contract_assessments": [],
              "confirmed_findings": [],
              "rejected_hypotheses": [],
              "uncertainties": [],
              "observation_refs": [],
              "investigation_summary": "Bad status.",
              "status": "done"
            }
            """
        )


def test_parse_reviewer_result_rejects_non_list_result_fields():
    with pytest.raises(ReviewerResultParseError, match="contract_assessments must be a list"):
        parse_reviewer_result(
            """
            {
              "contract_assessments": null,
              "confirmed_findings": [],
              "rejected_hypotheses": [],
              "uncertainties": [],
              "observation_refs": [],
              "investigation_summary": "Malformed list field.",
              "status": "failed"
            }
            """
        )


def test_parse_reviewer_result_rejects_non_list_evidence_refs():
    with pytest.raises(ReviewerResultParseError, match="contract assessment evidence_refs must be a list"):
        parse_reviewer_result(
            """
            {
              "contract_assessments": [
                {
                  "contract": "intent_alignment",
                  "status": "covered",
                  "summary": "The change matches the stated intent.",
                  "evidence_refs": null
                }
              ],
              "confirmed_findings": [],
              "rejected_hypotheses": [],
              "uncertainties": [],
              "observation_refs": [],
              "investigation_summary": "Malformed nested list field.",
              "status": "failed"
            }
            """
        )


def make_assignment() -> Assignment:
    return Assignment(
        role="Core Reviewer",
        mission="Check intent alignment",
        assignment_reason=["sensitive path changed: auth.py"],
        assigned_contract=["intent_alignment"],
        required_checks=["map changed behavior to intent"],
        initial_context=InitialContext(
            changed_files=["auth.py"],
            diff_ranges=["auth.py"],
            observation_refs=["O-diff-auth"],
        ),
        max_turns=6,
        max_tool_calls=12,
    )


def make_intent() -> IntentPacket:
    return IntentPacket(
        goal="Refactor admin check",
        sources={"goal": IntentSource.EXPLICIT},
        status=IntentStatus.PARTIAL,
        uncertainties=["acceptance criteria are not explicitly declared"],
    )


def test_run_single_reviewer_calls_adapter_and_parses_result():
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="""
                {
                  "contract_assessments": [],
                  "confirmed_findings": [],
                  "rejected_hypotheses": [],
                  "uncertainties": ["No tool gateway available."],
                  "observation_refs": ["O-diff-auth"],
                  "investigation_summary": "Reviewed diff excerpt.",
                  "status": "partial"
                }
                """,
                provider_name="fake",
                model="fake-reviewer",
            )
        ]
    )

    run = run_single_reviewer(
        adapter=adapter,
        assignment=make_assignment(),
        intent=make_intent(),
        diff_excerpt=["-    return user.role == 'admin'", "+    return True"],
        observations={"O-diff-auth": "auth.py changed between base and head"},
        trace_id="trace-reviewer-1",
    )

    assert run.result.status is ReviewerResultStatus.PARTIAL
    assert run.result.observation_refs == ["O-diff-auth"]
    assert run.response.provider_name == "fake-tool-calling"
    assert run.envelope.parameters["trace_id"] == "trace-reviewer-1"
    assert "return True" in run.envelope.messages[0]["content"]
    assert adapter.requests[0].tools == []
    assert adapter.requests[0].parameters["tool_choice"] == "none"


def test_run_single_reviewer_rejects_tool_calls_without_executing_tools():
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": "auth.py"})],
                provider_name="fake",
                model="fake-reviewer",
            )
        ]
    )

    run = run_single_reviewer(
        adapter=adapter,
        assignment=make_assignment(),
        intent=make_intent(),
        diff_excerpt=["+changed"],
        observations={},
        trace_id="trace-reviewer-tool-call",
    )

    assert run.result.status is ReviewerResultStatus.FAILED
    assert "single-shot reviewer received tool calls" in run.result.investigation_summary
    assert run.response.provider_name == "fake-tool-calling"


def test_run_single_reviewer_handles_invalid_adapter_response():
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error="bad response shape",
                provider_name="fake",
                model="fake-reviewer",
            )
        ]
    )

    run = run_single_reviewer(
        adapter=adapter,
        assignment=make_assignment(),
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="trace-reviewer-invalid",
    )

    assert run.result.status is ReviewerResultStatus.FAILED
    assert run.result.uncertainties == ["bad response shape"]
    assert run.response.content == "bad response shape"


def test_run_single_reviewer_handles_malformed_final_list_field():
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="""
                {
                  "contract_assessments": null,
                  "confirmed_findings": [],
                  "rejected_hypotheses": [],
                  "uncertainties": [],
                  "observation_refs": [],
                  "investigation_summary": "Malformed list field.",
                  "status": "failed"
                }
                """,
                provider_name="fake",
                model="fake-reviewer",
            )
        ]
    )

    run = run_single_reviewer(
        adapter=adapter,
        assignment=make_assignment(),
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="trace-reviewer-malformed-final",
    )

    assert run.result.status is ReviewerResultStatus.FAILED
    assert "single-shot final response parse failed" in run.result.uncertainties[0]
    assert "contract_assessments must be a list" in run.result.uncertainties[0]
    assert "single-shot final response parse failed" in run.result.investigation_summary


def test_run_single_reviewer_uses_diff_excerpt_as_code_snippet():
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="""
                {
                  "contract_assessments": [],
                  "confirmed_findings": [],
                  "rejected_hypotheses": [],
                  "uncertainties": [],
                  "observation_refs": [],
                  "investigation_summary": "Reviewed diff.",
                  "status": "completed"
                }
                """,
                provider_name="fake",
                model="fake-reviewer",
            )
        ]
    )

    run = run_single_reviewer(
        adapter=adapter,
        assignment=make_assignment(),
        intent=make_intent(),
        diff_excerpt=["+changed"],
        observations={},
        trace_id="trace-reviewer-2",
    )

    assert "Diff Excerpt" in run.envelope.messages[0]["content"]
    assert "+changed" in run.envelope.messages[0]["content"]
