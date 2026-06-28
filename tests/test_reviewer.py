import pytest

from review_agent.models import ReviewerResultStatus
from review_agent.reviewer import ReviewerResultParseError, parse_reviewer_result


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
