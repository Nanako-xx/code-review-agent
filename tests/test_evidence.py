from review_agent.evidence import reconcile_evidence, reconciliation_to_dict
from review_agent.model_protocol import ModelResponse
from review_agent.models import ReviewerFinding, ReviewerResult, ReviewerResultStatus
from review_agent.orchestrator import ReviewerExecution
from tests.test_orchestrator import make_assignment


def execution(index, role, findings):
    assignment = make_assignment(role)
    return ReviewerExecution(
        reviewer_index=index,
        trace_id=f"review-1-reviewer-{index}",
        assignment=assignment,
        envelope=None,
        response=ModelResponse(content="{}", provider_name="fake", model="fake"),
        result=ReviewerResult(
            confirmed_findings=findings,
            investigation_summary=f"{role} done",
            status=ReviewerResultStatus.COMPLETED,
        ),
    )


def finding(claim, refs):
    return ReviewerFinding(
        claim=claim,
        severity="high",
        confidence="high",
        evidence_refs=refs,
        suggested_action="fix it",
    )


def test_reconcile_evidence_rejects_findings_with_missing_evidence_refs():
    reconciliation = reconcile_evidence(
        executions=[execution(0, "Core Reviewer", [finding("Auth bypass", ["O-known", "O-missing"])])],
        authorized_observation_ids={"O-known"},
    )

    assert reconciliation.canonical_findings == []
    assert len(reconciliation.rejected_findings) == 1
    rejected = reconciliation.rejected_findings[0]
    assert rejected.reason == "unsupported_claim"
    assert rejected.missing_evidence_refs == ["O-missing"]
    assert reconciliation.evidence_quality == "unsupported_claims"


def test_reconcile_evidence_keeps_and_deduplicates_supported_findings():
    reconciliation = reconcile_evidence(
        executions=[
            execution(0, "Core Reviewer", [finding("Auth bypass", ["O-auth"])]),
            execution(1, "Adversarial Reviewer", [finding(" auth bypass ", ["O-auth"])]),
        ],
        authorized_observation_ids={"O-auth"},
    )

    payload = reconciliation_to_dict(reconciliation)

    assert payload["evidence_quality"] == "verified"
    assert len(payload["canonical_findings"]) == 1
    assert payload["canonical_findings"][0]["claim"] == "Auth bypass"
    assert payload["canonical_findings"][0]["reviewer_indices"] == [0, 1]
    assert payload["canonical_findings"][0]["roles"] == ["Core Reviewer", "Adversarial Reviewer"]
