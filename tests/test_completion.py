from review_agent.completion import check_completion, completion_to_dict
from review_agent.evidence import EvidenceReconciliation
from review_agent.models import IntentPacket, IntentSource, IntentStatus, ReviewerResult, ReviewerResultStatus
from review_agent.orchestrator import ReviewerExecution
from review_agent.provider import ModelResponse
from tests.test_orchestrator import make_assignment


def execution(index, role, status):
    return ReviewerExecution(
        reviewer_index=index,
        trace_id=f"review-1-reviewer-{index}",
        assignment=make_assignment(role),
        envelope=None,
        response=ModelResponse(content="{}", provider_name="fake", model="fake"),
        result=ReviewerResult(status=status, investigation_summary=f"{role} {status.value}"),
    )


def intent(status=IntentStatus.SUFFICIENT):
    return IntentPacket(goal="Review change", sources={"goal": IntentSource.EXPLICIT}, status=status)


def reconciliation(canonical=0, rejected=0):
    return EvidenceReconciliation(
        canonical_findings=[object()] * canonical,
        rejected_findings=[object()] * rejected,
        remaining_disagreements=[],
        contract_coverage=[],
        evidence_quality="verified" if rejected == 0 else "unsupported_claims",
    )


def test_completion_blocks_when_core_reviewer_failed():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[
            execution(0, "Core Reviewer", ReviewerResultStatus.FAILED),
            execution(1, "Adversarial Reviewer", ReviewerResultStatus.COMPLETED),
        ],
        reconciliation=reconciliation(),
    )

    assert result.status == "blocked"
    assert result.recommendation == "manual_review"
    assert "Core Reviewer failed" in result.blockers


def test_completion_with_uncertainties_when_specialist_failed():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[
            execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED),
            execution(1, "Adversarial Reviewer", ReviewerResultStatus.FAILED),
        ],
        reconciliation=reconciliation(),
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert result.missing_perspectives == ["Adversarial Reviewer"]


def test_completion_requires_manual_review_for_unsupported_findings():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)],
        reconciliation=reconciliation(rejected=1),
    )

    payload = completion_to_dict(result)

    assert payload["status"] == "completed_with_uncertainties"
    assert payload["recommendation"] == "manual_review"
    assert "unsupported findings rejected" in payload["uncertainties"]


def test_completion_with_uncertainties_when_reviewer_is_partial():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.PARTIAL)],
        reconciliation=reconciliation(),
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert "Core Reviewer returned partial review" in result.uncertainties
