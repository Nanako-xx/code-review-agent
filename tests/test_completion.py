from review_agent.completion import check_completion, completion_to_dict
from review_agent.evidence import ContractCoverage, EvidenceReconciliation
from review_agent.model_protocol import ModelResponse
from review_agent.models import IntentPacket, IntentSource, IntentStatus, ReviewerResult, ReviewerResultStatus
from review_agent.orchestrator import ReviewerExecution
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


def reconciliation_with_coverage(*coverage_rows):
    return EvidenceReconciliation(
        canonical_findings=[],
        rejected_findings=[],
        remaining_disagreements=[],
        contract_coverage=list(coverage_rows),
        evidence_quality="verified",
    )


def coverage(index, role, contract="regression_safety", status="covered"):
    return ContractCoverage(
        reviewer_index=index,
        role=role,
        contract=contract,
        status=status,
        summary=f"{role} covered {contract}",
        evidence_refs=[],
        unsupported_evidence_refs=[],
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


def test_completion_blocks_when_core_reviewer_did_not_run():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[],
        reconciliation=reconciliation(),
    )

    assert result.status == "blocked"
    assert result.recommendation == "manual_review"
    assert result.blockers == ["Core Reviewer did not run"]


def test_completion_with_uncertainties_when_specialist_failed():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[
            execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED),
            execution(1, "Adversarial Reviewer", ReviewerResultStatus.FAILED),
        ],
        reconciliation=reconciliation_with_coverage(coverage(0, "Core Reviewer")),
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert result.missing_perspectives == ["Adversarial Reviewer"]


def test_completion_requires_manual_review_for_unsupported_findings():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)],
        reconciliation=EvidenceReconciliation(
            canonical_findings=[],
            rejected_findings=[object()],
            remaining_disagreements=[],
            contract_coverage=[coverage(0, "Core Reviewer")],
            evidence_quality="unsupported_claims",
        ),
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


def test_completion_blocks_when_core_contract_coverage_missing():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)],
        reconciliation=reconciliation_with_coverage(),
    )

    assert result.status == "blocked"
    assert result.recommendation == "manual_review"
    assert "Core Reviewer missing contract coverage: regression_safety" in result.blockers


def test_completion_with_uncertainties_when_specialist_contract_coverage_missing():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[
            execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED),
            execution(1, "Adversarial Reviewer", ReviewerResultStatus.COMPLETED),
        ],
        reconciliation=reconciliation_with_coverage(coverage(0, "Core Reviewer")),
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert "Adversarial Reviewer missing contract coverage: regression_safety" in result.uncertainties


def test_completion_with_uncertainties_when_intent_is_partial():
    result = check_completion(
        intent=IntentPacket(
            goal="Review change",
            sources={"goal": IntentSource.INFERRED},
            status=IntentStatus.PARTIAL,
            uncertainties=["acceptance criteria unclear"],
        ),
        quality_results=[],
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)],
        reconciliation=reconciliation_with_coverage(coverage(0, "Core Reviewer")),
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert "Intent Packet partial" in result.uncertainties
    assert "acceptance criteria unclear" in result.uncertainties


def test_completion_records_blocked_non_core_reviewer_as_missing_perspective():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[
            execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED),
            execution(1, "Adversarial Reviewer", ReviewerResultStatus.BLOCKED),
        ],
        reconciliation=reconciliation_with_coverage(coverage(0, "Core Reviewer")),
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert result.missing_perspectives == ["Adversarial Reviewer"]


def test_completion_blocks_when_core_contract_coverage_is_unknown():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)],
        reconciliation=reconciliation_with_coverage(coverage(0, "Core Reviewer", status="unknown")),
    )

    assert result.status == "blocked"
    assert result.recommendation == "manual_review"
    assert "Core Reviewer incomplete contract coverage: regression_safety" in result.blockers


def test_completion_with_uncertainties_when_non_core_contract_coverage_is_partial():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[
            execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED),
            execution(1, "Adversarial Reviewer", ReviewerResultStatus.COMPLETED),
        ],
        reconciliation=reconciliation_with_coverage(
            coverage(0, "Core Reviewer"),
            coverage(1, "Adversarial Reviewer", status="partial"),
        ),
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert "Adversarial Reviewer incomplete contract coverage: regression_safety" in result.uncertainties


def test_completion_blocks_when_final_risk_is_required_but_missing():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)],
        reconciliation=reconciliation_with_coverage(coverage(0, "Core Reviewer")),
        require_final_risk=True,
    )

    assert result.status == "blocked"
    assert result.recommendation == "manual_review"
    assert "Final risk reassessment not completed" in result.blockers


def test_completion_requires_manual_review_when_final_risk_is_high():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)],
        reconciliation=reconciliation_with_coverage(coverage(0, "Core Reviewer")),
        require_final_risk=True,
        final_risk_level="high",
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert "Final risk is high" in result.uncertainties
