from __future__ import annotations

import json

import pytest

from review_agent.evidence import ConflictHint, FindingCandidate, ReconciliationPrepass
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_adapter_factory import (
    ModelAdapterConfig,
    build_model_adapter_factory_from_config,
)
from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse
from review_agent.observations import Observation
from review_agent.reconciler import (
    SemanticProposalParseError,
    batch_reconciliation_packet,
    build_reconciliation_packet,
    compile_semantic_proposals,
    parse_semantic_proposal,
    reconcile_semantically,
    semantic_reconciliation_from_dict,
    semantic_reconciliation_to_dict,
)


BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40


def _candidate(
    suffix: str,
    *,
    claim: str,
    severity: str = "medium",
    confidence: str = "medium",
    evidence_ref: str | None = None,
) -> FindingCandidate:
    ref = evidence_ref or f"O-{suffix}"
    return FindingCandidate(
        finding_id=f"F-{suffix}",
        origin="initial",
        reviewer_task_id=f"reviewer-{suffix}",
        reviewer_index=int(suffix) if suffix.isdigit() else 0,
        role=f"Reviewer {suffix}",
        role_kind="core",
        claim=claim,
        severity=severity,
        confidence=confidence,
        path="app.py",
        line=10,
        impact="Behavior may be incorrect.",
        suggested_action="Correct the behavior.",
        verification_performed=["read relevant code"],
        evidence_refs=[ref],
        validation_status="supported",
    )


def _observation(observation_id: str, *, source: str = "read_range") -> Observation:
    return Observation(
        observation_id=observation_id,
        source=source,
        revision=f"head@{HEAD_SHA}",
        path="app.py",
        line_start=1,
        line_end=20,
        content_hash="c" * 64,
        raw_artifact_ref=f"observations/{observation_id}.txt",
        context_view=f"verified context for {observation_id}",
    )


def _prepass(
    candidates: list[FindingCandidate],
    *,
    hints: list[ConflictHint] | None = None,
) -> ReconciliationPrepass:
    return ReconciliationPrepass(
        review_id="review-semantic",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        candidate_catalog={item.finding_id: item for item in candidates},
        conflict_hints=list(hints or []),
        rejected_findings=[],
        contract_coverage=[],
        evidence_quality="verified",
    )


def _packet(
    candidates: list[FindingCandidate],
    *,
    hints: list[ConflictHint] | None = None,
):
    prepass = _prepass(candidates, hints=hints)
    observations = {
        ref: _observation(ref)
        for candidate in candidates
        for ref in candidate.evidence_refs
    }
    return prepass, observations, build_reconciliation_packet(prepass, observations)


def _proposal_payload(
    candidates: list[FindingCandidate],
    *,
    groups: list[dict[str, object]] | None = None,
    rejections: list[dict[str, object]] | None = None,
    disagreements: list[dict[str, object]] | None = None,
    supplemental_requests: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if groups is None:
        groups = [
            {
                "member_ids": [candidate.finding_id],
                "representative_id": candidate.finding_id,
                "canonical_claim": candidate.claim,
                "rationale": "The cited Observation supports this candidate.",
                "supporting_refs": list(candidate.evidence_refs),
                "proposed_confidence": candidate.confidence,
            }
            for candidate in candidates
        ]
    return {
        "canonical_groups": groups,
        "rejections": list(rejections or []),
        "disagreements": list(disagreements or []),
        "supplemental_requests": list(supplemental_requests or []),
        "uncertainties": [],
        "summary": "Semantic reconciliation completed.",
    }


def test_packet_batching_is_deterministic_and_keeps_conflict_components_together():
    candidates = [
        _candidate("0", claim="First claim"),
        _candidate("1", claim="Related claim"),
        _candidate("2", claim="Independent claim"),
    ]
    hint = ConflictHint(
        conflict_id="C-related",
        candidate_ids=["F-0", "F-1"],
        kind="same_location",
        summary="Candidates concern the same location.",
    )
    _, _, packet = _packet(candidates, hints=[hint])

    first = batch_reconciliation_packet(packet, max_candidates_per_batch=1)
    second = batch_reconciliation_packet(packet, max_candidates_per_batch=1)

    assert first == second
    assert first[0].candidate_ids == ("F-0", "F-1")
    assert first[1].candidate_ids == ("F-2",)
    assert all(batch.input_digest != "0" * 64 for batch in first)


def test_parser_accepts_complete_candidate_accounting():
    candidates = [_candidate("0", claim="Authorization check can be bypassed")]
    _, _, packet = _packet(candidates)

    proposal = parse_semantic_proposal(
        json.dumps(_proposal_payload(candidates)),
        packet,
    )

    assert proposal.canonical_groups[0].member_ids == ("F-0",)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            '{"canonical_groups":[],"canonical_groups":[],"rejections":[],'
            '"disagreements":[],"supplemental_requests":[],"uncertainties":[],'
            '"summary":"duplicate"}',
            "duplicate field",
        ),
        (
            json.dumps(
                _proposal_payload(
                    [_candidate("0", claim="Candidate")],
                    groups=[],
                )
            ),
            "missing candidates",
        ),
    ],
)
def test_parser_rejects_duplicate_json_keys_and_incomplete_accounting(content, message):
    candidate = _candidate("0", claim="Candidate")
    _, _, packet = _packet([candidate])

    with pytest.raises(SemanticProposalParseError, match=message):
        parse_semantic_proposal(content, packet)


def test_parser_rejects_unknown_observation_references():
    candidate = _candidate("0", claim="Candidate")
    _, _, packet = _packet([candidate])
    payload = _proposal_payload([candidate])
    payload["canonical_groups"][0]["supporting_refs"] = ["O-invented"]

    with pytest.raises(SemanticProposalParseError, match="supporting_refs"):
        parse_semantic_proposal(json.dumps(payload), packet)


def test_runtime_preserves_supported_high_severity_rejection():
    candidate = _candidate(
        "0",
        claim="Authorization can be bypassed",
        severity="high",
        confidence="high",
    )
    prepass, _, packet = _packet([candidate])
    proposal = parse_semantic_proposal(
        json.dumps(
            _proposal_payload(
                [candidate],
                groups=[],
                rejections=[
                    {
                        "candidate_id": "F-0",
                        "reason": "unsupported_claim",
                        "rationale": "The model considered the support insufficient.",
                        "decision_refs": [],
                    }
                ],
            )
        ),
        packet,
    )

    reconciliation, _ = compile_semantic_proposals(prepass, packet, [proposal])

    assert [item.claim for item in reconciliation.canonical_findings] == [
        "Authorization can be bypassed"
    ]
    assert reconciliation.rejected_findings == ()
    assert reconciliation.remaining_disagreements[0].decision_source == "runtime_policy"
    assert reconciliation.policy_actions == ("preserved_severe_finding:F-0",)


def test_reconciler_retries_malformed_responses_then_falls_back_conservatively():
    candidate = _candidate("0", claim="Candidate", severity="high")
    prepass, observations, _ = _packet([candidate])
    adapter = FakeToolCallingAdapter(
        [
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="not-json",
            ),
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps({"unexpected": True}),
            ),
        ]
    )

    run = reconcile_semantically(
        prepass,
        observations,
        adapter=adapter,
        max_provider_attempts=2,
    )

    assert run.status == "fallback"
    assert run.reconciliation.model.status == "fallback"
    assert [attempt.status for attempt in run.batches[0].attempts] == [
        "parse_error",
        "parse_error",
    ]
    assert [item.claim for item in run.reconciliation.canonical_findings] == [
        "Candidate"
    ]
    assert run.supplemental_requests == ()


def test_local_only_reconciliation_is_deterministic_and_does_not_invoke_a_model():
    candidates = [
        _candidate("0", claim="Same claim", evidence_ref="O-shared"),
        _candidate("1", claim=" same claim ", evidence_ref="O-shared"),
    ]
    prepass, observations, _ = _packet(candidates)

    run = reconcile_semantically(prepass, observations, adapter=None)

    assert run.status == "local_only"
    assert run.batches == ()
    assert len(run.reconciliation.canonical_findings) == 1
    assert run.reconciliation.model.status == "disabled"


def test_fake_provider_produces_an_accepted_semantic_result():
    candidate = _candidate("0", claim="Candidate")
    prepass, observations, _ = _packet([candidate])
    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="fake",
            model=None,
            base_url=None,
            api_key_env="REVIEW_AGENT_API_KEY",
        )
    )

    run = reconcile_semantically(prepass, observations, adapter=factory.create())

    assert run.status == "accepted"
    assert run.reconciliation.model.status == "accepted"
    assert run.batches[0].provider_name == "fake"


def test_semantic_reconciliation_round_trips_and_rejects_schema_drift():
    candidate = _candidate("0", claim="Candidate")
    prepass, observations, _ = _packet([candidate])
    reconciliation = reconcile_semantically(
        prepass,
        observations,
        adapter=None,
    ).reconciliation
    payload = semantic_reconciliation_to_dict(reconciliation)

    assert semantic_reconciliation_from_dict(payload) == reconciliation

    drifted = dict(payload)
    drifted["invented"] = True
    with pytest.raises(ValueError, match="exact fields"):
        semantic_reconciliation_from_dict(drifted)
