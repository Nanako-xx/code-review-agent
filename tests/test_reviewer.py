import pytest

from review_agent.models import Assignment, InitialContext, IntentPacket, IntentSource, IntentStatus, ReviewerResultStatus
from review_agent.provider import FakeProvider
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


def test_run_single_reviewer_calls_provider_and_parses_result():
    provider = FakeProvider(
        """
        {
          "contract_assessments": [],
          "confirmed_findings": [],
          "rejected_hypotheses": [],
          "uncertainties": ["No tool gateway available."],
          "observation_refs": ["O-diff-auth"],
          "investigation_summary": "Reviewed diff excerpt.",
          "status": "partial"
        }
        """
    )

    run = run_single_reviewer(
        provider=provider,
        assignment=make_assignment(),
        intent=make_intent(),
        diff_excerpt=["-    return user.role == 'admin'", "+    return True"],
        observations={"O-diff-auth": "auth.py changed between base and head"},
        trace_id="trace-reviewer-1",
    )

    assert run.result.status is ReviewerResultStatus.PARTIAL
    assert run.result.observation_refs == ["O-diff-auth"]
    assert run.response.provider_name == "fake"
    assert run.envelope.parameters["trace_id"] == "trace-reviewer-1"
    assert "return True" in run.envelope.messages[0]["content"]


def test_run_single_reviewer_uses_diff_excerpt_as_code_snippet():
    provider = FakeProvider(
        """
        {
          "contract_assessments": [],
          "confirmed_findings": [],
          "rejected_hypotheses": [],
          "uncertainties": [],
          "observation_refs": [],
          "investigation_summary": "Reviewed diff.",
          "status": "completed"
        }
        """
    )

    run = run_single_reviewer(
        provider=provider,
        assignment=make_assignment(),
        intent=make_intent(),
        diff_excerpt=["+changed"],
        observations={},
        trace_id="trace-reviewer-2",
    )

    assert "Diff Excerpt" in run.envelope.messages[0]["content"]
    assert "+changed" in run.envelope.messages[0]["content"]
