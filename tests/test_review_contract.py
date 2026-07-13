from dataclasses import FrozenInstanceError, replace

import pytest

from review_agent.models import (
    Assignment,
    ContractAssessment,
    ContractItemStatus,
    InitialContext,
    ReviewerFinding,
    ReviewerResult,
    ReviewerResultStatus,
)
from review_agent.review_contract import ReviewerCompletionValidation, validate_reviewer_completion


def assignment(*contracts: str) -> Assignment:
    return Assignment(
        role="core",
        mission="Review the change",
        assignment_reason=[],
        assigned_contract=list(contracts),
        required_checks=[],
        initial_context=InitialContext(),
        max_turns=4,
        max_tool_calls=8,
    )


def assessment(
    contract: str,
    status: ContractItemStatus = ContractItemStatus.COVERED,
    *evidence_refs: str,
) -> ContractAssessment:
    return ContractAssessment(
        contract=contract,
        status=status,
        summary=f"Assessed {contract}",
        evidence_refs=list(evidence_refs),
    )


def finding(*evidence_refs: str) -> ReviewerFinding:
    return ReviewerFinding(
        claim="A defect is present",
        severity="high",
        confidence="high",
        evidence_refs=list(evidence_refs),
        suggested_action="Fix the defect.",
        path="app.py",
        line=1,
        impact="The changed behavior can return an incorrect result.",
        verification_performed=["Compared base and head behavior."],
    )


def test_completed_result_with_complete_contract_is_accepted():
    result = ReviewerResult(
        contract_assessments=[
            assessment("correctness", ContractItemStatus.COVERED, "O-1"),
            assessment("regression_safety", ContractItemStatus.NOT_APPLICABLE),
        ],
        confirmed_findings=[finding("O-1")],
        observation_refs=["O-1"],
        investigation_summary="Reviewed the assigned contracts and evidence.",
        status=ReviewerResultStatus.COMPLETED,
    )

    validation = validate_reviewer_completion(
        assignment("correctness", "regression_safety"),
        result,
        {"O-1"},
    )

    assert validation == ReviewerCompletionValidation(accepted=True, deficiencies=())


def test_completed_result_rejects_missing_contract_assessment():
    result = ReviewerResult(
        investigation_summary="Review complete.",
        status=ReviewerResultStatus.COMPLETED,
    )

    validation = validate_reviewer_completion(assignment("correctness"), result, set())

    assert validation == ReviewerCompletionValidation(
        accepted=False,
        deficiencies=("missing contract assessment: correctness",),
    )


def test_completed_result_rejects_duplicate_contract_assessments():
    result = ReviewerResult(
        contract_assessments=[assessment("correctness"), assessment("correctness")],
        investigation_summary="Review complete.",
        status=ReviewerResultStatus.COMPLETED,
    )

    validation = validate_reviewer_completion(assignment("correctness"), result, set())

    assert validation == ReviewerCompletionValidation(
        accepted=False,
        deficiencies=("duplicate contract assessment: correctness",),
    )


@pytest.mark.parametrize("status", [ContractItemStatus.PARTIAL, ContractItemStatus.UNKNOWN])
def test_completed_result_rejects_non_final_contract_status(status: ContractItemStatus):
    result = ReviewerResult(
        contract_assessments=[assessment("correctness", status)],
        investigation_summary="Review complete.",
        status=ReviewerResultStatus.COMPLETED,
    )

    validation = validate_reviewer_completion(assignment("correctness"), result, set())

    assert validation == ReviewerCompletionValidation(
        accepted=False,
        deficiencies=(f"incomplete contract assessment: correctness ({status.value})",),
    )


@pytest.mark.parametrize("summary", ["", " \t\n"])
def test_completed_result_rejects_empty_investigation_summary(summary: str):
    result = ReviewerResult(
        contract_assessments=[assessment("correctness")],
        investigation_summary=summary,
        status=ReviewerResultStatus.COMPLETED,
    )

    validation = validate_reviewer_completion(assignment("correctness"), result, set())

    assert validation == ReviewerCompletionValidation(
        accepted=False,
        deficiencies=("completed result has an empty investigation summary",),
    )


@pytest.mark.parametrize("status", list(ReviewerResultStatus))
def test_every_result_status_rejects_finding_without_evidence(status: ReviewerResultStatus):
    result = ReviewerResult(
        contract_assessments=[assessment("correctness")],
        confirmed_findings=[finding()],
        investigation_summary="Review complete.",
        status=status,
    )

    validation = validate_reviewer_completion(assignment("correctness"), result, set())

    assert validation == ReviewerCompletionValidation(
        accepted=False,
        deficiencies=("confirmed finding 0 has no evidence refs",),
    )


def test_unknown_evidence_deficiencies_are_ordered_and_deduplicated():
    result = ReviewerResult(
        contract_assessments=[
            assessment("correctness", ContractItemStatus.COVERED, "O-contract", "O-contract")
        ],
        confirmed_findings=[finding("O-finding", "O-finding")],
        observation_refs=["O-result", "O-result", "O-result-2"],
        investigation_summary="Review complete.",
        status=ReviewerResultStatus.COMPLETED,
    )

    validation = validate_reviewer_completion(assignment("correctness"), result, set())

    assert validation == ReviewerCompletionValidation(
        accepted=False,
        deficiencies=(
            "unauthorized result observation ref: O-result",
            "unauthorized result observation ref: O-result-2",
            "unauthorized finding evidence ref: O-finding",
            "unauthorized contract assessment evidence ref: O-contract",
        ),
    )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"claim": ""}, "confirmed finding 0 has an empty claim"),
        ({"severity": "urgent"}, "confirmed finding 0 has invalid severity: urgent"),
        ({"confidence": "certain"}, "confirmed finding 0 has invalid confidence: certain"),
        ({"path": None}, "confirmed finding 0 has no path"),
        (
            {"path": "../app.py"},
            "confirmed finding 0 has unsafe path: ../app.py",
        ),
        ({"line": 0}, "confirmed finding 0 has no positive line"),
        ({"impact": ""}, "confirmed finding 0 has no impact"),
        ({"suggested_action": None}, "confirmed finding 0 has no suggested action"),
        (
            {"verification_performed": []},
            "confirmed finding 0 has no valid verification performed",
        ),
    ],
)
def test_finding_schema_is_enforced(changes: dict[str, object], expected: str):
    invalid_finding = replace(finding("O-1"), **changes)
    result = ReviewerResult(
        contract_assessments=[
            assessment("correctness", ContractItemStatus.COVERED, "O-1")
        ],
        confirmed_findings=[invalid_finding],
        observation_refs=["O-1"],
        investigation_summary="Reviewed the finding.",
        status=ReviewerResultStatus.COMPLETED,
    )

    validation = validate_reviewer_completion(
        assignment("correctness"),
        result,
        {"O-1"},
    )

    assert expected in validation.deficiencies


def test_partial_result_may_omit_contract_assessments_and_summary():
    result = ReviewerResult(status=ReviewerResultStatus.PARTIAL)

    validation = validate_reviewer_completion(assignment("correctness"), result, set())

    assert validation == ReviewerCompletionValidation(accepted=True, deficiencies=())


@pytest.mark.parametrize(
    "status",
    [ReviewerResultStatus.PARTIAL, ReviewerResultStatus.BLOCKED, ReviewerResultStatus.FAILED],
)
def test_non_completed_result_still_rejects_unknown_evidence(status: ReviewerResultStatus):
    result = ReviewerResult(
        observation_refs=["O-unknown"],
        status=status,
    )

    validation = validate_reviewer_completion(assignment("correctness"), result, set())

    assert validation == ReviewerCompletionValidation(
        accepted=False,
        deficiencies=("unauthorized result observation ref: O-unknown",),
    )


def test_validation_result_is_frozen():
    validation = ReviewerCompletionValidation(accepted=True, deficiencies=())

    with pytest.raises(FrozenInstanceError):
        validation.accepted = False  # type: ignore[misc]
