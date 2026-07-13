import pytest

from review_agent.evidence import (
    build_reconciliation_prepass,
    reconcile_evidence,
    reconciliation_to_dict,
)
from review_agent.model_protocol import ModelResponse
from review_agent.models import ReviewerFinding, ReviewerResult, ReviewerResultStatus
from review_agent.observations import Observation
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


def observation(
    observation_id,
    *,
    revision,
    path="auth.py",
    line_start=10,
    line_end=20,
    source="read_range",
):
    return Observation(
        observation_id=observation_id,
        source=source,
        revision=revision,
        path=path,
        line_start=line_start,
        line_end=line_end,
        content_hash="a" * 64,
        raw_artifact_ref=f"observations/{observation_id}.txt",
        context_view=f"context for {observation_id}",
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


def test_prepass_keeps_stable_candidates_before_exact_deduplication():
    head = "b" * 40
    executions = [
        execution(
            0,
            "Core Reviewer",
            [
                ReviewerFinding(
                    claim="Auth bypass",
                    severity="high",
                    confidence="high",
                    evidence_refs=["O-auth"],
                    path="auth.py",
                    line=12,
                    impact="Unauthenticated access",
                    suggested_action="Enforce authorization",
                    verification_performed=["read auth guard"],
                )
            ],
        ),
        execution(
            1,
            "Adversarial Reviewer",
            [
                ReviewerFinding(
                    claim=" auth bypass ",
                    severity="high",
                    confidence="high",
                    evidence_refs=["O-auth"],
                    path="auth.py",
                    line=12,
                    impact="Unauthenticated access",
                    suggested_action="Enforce authorization",
                    verification_performed=["read auth guard"],
                )
            ],
        ),
    ]
    observations = {
        "O-auth": observation("O-auth", revision=f"head@{head}"),
    }

    first = build_reconciliation_prepass(
        executions,
        observations,
        review_id="review-1",
        base_sha="a" * 40,
        head_sha=head,
    )
    reordered = build_reconciliation_prepass(
        list(reversed(executions)),
        observations,
        review_id="review-1",
        base_sha="a" * 40,
        head_sha=head,
    )

    assert len(first.candidate_catalog) == 2
    assert list(first.candidate_catalog) == list(reordered.candidate_catalog)
    assert all(candidate.validation_status == "supported" for candidate in first.candidate_catalog.values())
    exact_hints = [hint for hint in first.conflict_hints if hint.kind == "exact_duplicate"]
    assert len(exact_hints) == 1
    assert exact_hints[0].candidate_ids == sorted(first.candidate_catalog)
    assert exact_hints[0].conflict_id.startswith("C-")


def test_prepass_rejects_unknown_stale_and_location_mismatched_evidence():
    base = "a" * 40
    head = "b" * 40
    executions = [
        execution(
            0,
            "Core Reviewer",
            [
                ReviewerFinding(
                    claim="Unknown evidence",
                    severity="medium",
                    confidence="low",
                    evidence_refs=["O-missing"],
                    path="auth.py",
                    line=12,
                ),
                ReviewerFinding(
                    claim="Stale evidence",
                    severity="medium",
                    confidence="low",
                    evidence_refs=["O-stale"],
                    path="auth.py",
                    line=12,
                ),
                ReviewerFinding(
                    claim="Wrong location",
                    severity="medium",
                    confidence="low",
                    evidence_refs=["O-other"],
                    path="auth.py",
                    line=12,
                ),
            ],
        )
    ]
    observations = {
        "O-stale": observation("O-stale", revision=f"head@{'c' * 40}"),
        "O-other": observation(
            "O-other",
            revision=f"head@{head}",
            path="other.py",
            line_start=1,
            line_end=5,
        ),
    }

    prepass = build_reconciliation_prepass(
        executions,
        observations,
        review_id="review-1",
        base_sha=base,
        head_sha=head,
    )

    reasons = {
        candidate.claim: candidate.deterministic_rejection_reason
        for candidate in prepass.candidate_catalog.values()
    }
    assert reasons == {
        "Stale evidence": "stale_evidence",
        "Unknown evidence": "unsupported_claim",
        "Wrong location": "unsupported_claim",
    }
    assert all(candidate.validation_status == "rejected" for candidate in prepass.candidate_catalog.values())
    assert prepass.evidence_quality == "degraded"


def test_prepass_emits_same_location_severity_and_location_conflict_hints():
    head = "b" * 40
    executions = [
        execution(
            0,
            "Core Reviewer",
            [
                ReviewerFinding(
                    claim="Authorization bypass",
                    severity="high",
                    confidence="high",
                    evidence_refs=["O-auth"],
                    path="auth.py",
                    line=12,
                )
            ],
        ),
        execution(
            1,
            "Adversarial Reviewer",
            [
                ReviewerFinding(
                    claim="Authorization bypass",
                    severity="medium",
                    confidence="medium",
                    evidence_refs=["O-auth-file"],
                    path="auth.py",
                    line=18,
                ),
                ReviewerFinding(
                    claim="Session fixation",
                    severity="medium",
                    confidence="medium",
                    evidence_refs=["O-auth"],
                    path="auth.py",
                    line=12,
                ),
            ],
        ),
    ]
    observations = {
        "O-auth": observation("O-auth", revision=f"head@{head}"),
        "O-auth-file": observation(
            "O-auth-file",
            revision=f"head@{head}",
            line_start=None,
            line_end=None,
            source="compare_base_head",
        ),
    }

    prepass = build_reconciliation_prepass(
        executions,
        observations,
        review_id="review-1",
        base_sha="a" * 40,
        head_sha=head,
    )

    kinds = {hint.kind for hint in prepass.conflict_hints}
    assert {"same_location", "shared_evidence", "severity_mismatch", "location_mismatch"} <= kinds
    assert len({hint.conflict_id for hint in prepass.conflict_hints}) == len(prepass.conflict_hints)


def test_prepass_preserves_initial_and_supplemental_execution_origins():
    head = "b" * 40
    initial = execution(0, "Core Reviewer", [finding("Initial finding", ["O-initial"])])
    supplemental = execution(
        1,
        "Targeted Reviewer",
        [finding("Supplemental finding", ["O-supplemental"])],
    )
    observations = {
        "O-initial": observation("O-initial", revision=f"head@{head}"),
        "O-supplemental": observation("O-supplemental", revision=f"head@{head}"),
    }

    prepass = build_reconciliation_prepass(
        [initial, supplemental],
        observations,
        review_id="review-1",
        base_sha="a" * 40,
        head_sha=head,
        execution_metadata_by_trace_id={
            initial.trace_id: {"origin": "initial", "task_id": "A-initial"},
            supplemental.trace_id: {
                "origin": "supplemental",
                "task_id": "STASK-" + "d" * 64,
            },
        },
    )

    by_claim = {candidate.claim: candidate for candidate in prepass.candidate_catalog.values()}
    assert by_claim["Initial finding"].origin == "initial"
    assert by_claim["Initial finding"].reviewer_task_id == "A-initial"
    assert by_claim["Supplemental finding"].origin == "supplemental"
    assert by_claim["Supplemental finding"].reviewer_task_id == "STASK-" + "d" * 64


def test_prepass_rejects_execution_metadata_for_an_unknown_trace():
    reviewer = execution(0, "Core Reviewer", [])

    with pytest.raises(ValueError, match="unknown trace IDs"):
        build_reconciliation_prepass(
            [reviewer],
            {},
            review_id="review-1",
            base_sha="a" * 40,
            head_sha="b" * 40,
            execution_metadata_by_trace_id={
                "unknown": {"origin": "initial", "task_id": "A-unknown"}
            },
        )
