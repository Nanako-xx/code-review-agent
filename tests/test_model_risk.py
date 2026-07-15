from __future__ import annotations

from dataclasses import replace
import json

import pytest

from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse
from review_agent.model_risk import (
    MAX_RISK_REASONS,
    RiskProposal,
    RiskProposalParseError,
    build_local_risk_run,
    compile_risk_proposal,
    parse_risk_proposal,
    risk_packet_to_model_input,
    risk_input_digest,
    risk_invocation_id,
    risk_model_run_to_dict,
    run_model_risk_assessment,
)
from review_agent.models import (
    CompiledRiskFloor,
    IntentStatus,
    MemoryReference,
    MemoryRiskSignal,
    RiskAssessment,
    RiskAssessmentPacket,
    RiskMemoryProjection,
    RiskLevel,
)


def _dimensions(prefix: str = "model") -> dict[str, str]:
    return {
        "impact": f"{prefix} impact",
        "blast_radius": f"{prefix} blast radius",
        "reversibility": f"{prefix} reversibility",
        "uncertainty": f"{prefix} uncertainty",
        "verification_strength": f"{prefix} verification strength",
    }


def _proposal_payload(
    *,
    level: str = "high",
    signal_refs: list[str] | None = None,
) -> dict[str, object]:
    return {
        "level": level,
        "dimensions": _dimensions(),
        "reasons": ["public behavior may change"],
        "signal_refs": list(signal_refs or ["changed_file:app.py"]),
        "uncertainties": ["caller behavior is not fully known"],
        "suggested_focus": ["caller compatibility"],
    }


def _packet(*, quality_gates: dict[str, str] | None = None) -> RiskAssessmentPacket:
    gates = quality_gates or {"python_compile": "passed"}
    return RiskAssessmentPacket(
        change_summary={
            "repository_path": "C:/repo",
            "base_revision": "main",
            "head_revision": "HEAD",
            "changed_files": ["app.py"],
            "diff_stat": "1 file changed",
        },
        deterministic_signals={
            "quality_gates": gates,
            "changed_file_count": 1,
        },
        intent_status=IntentStatus.PARTIAL,
        intent_uncertainties=["acceptance criteria are not explicit"],
        diff_excerpt=["+def changed():"],
        changed_symbols=[
            {
                "path": "app.py",
                "qualified_name": "changed",
                "kind": "function",
                "change_type": "added",
                "line_start": 1,
                "line_end": 1,
            }
        ],
        signal_catalog={
            "changed_file:app.py": "Changed file: app.py",
            "quality_gate:python_compile": "Quality gate python_compile: passed",
        },
    )


def _local(level: RiskLevel = RiskLevel.MEDIUM) -> RiskAssessment:
    return RiskAssessment(
        level=level,
        dimensions=_dimensions("local"),
        reasons=["local reason"],
        signal_refs=["quality_gate:python_compile"],
        uncertainties=["local uncertainty"],
        suggested_focus=["local focus"],
    )


def _final(payload: dict[str, object], *, raw_id: int = 1) -> ModelTurnResponse:
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(payload),
        raw={"response_id": raw_id},
        model="risk-model",
    )


def test_parse_risk_proposal_accepts_exact_five_dimensions_and_authorized_refs():
    proposal = parse_risk_proposal(
        json.dumps(_proposal_payload()),
        _packet(),
    )

    assert proposal.level is RiskLevel.HIGH
    assert list(proposal.dimensions) == [
        "impact",
        "blast_radius",
        "reversibility",
        "uncertainty",
        "verification_strength",
    ]
    assert proposal.signal_refs == ["changed_file:app.py"]


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda value: value.update({"evidence_refs": []}), "unsupported field"),
        (
            lambda value: value["dimensions"].pop("reversibility"),
            "missing required field",
        ),
        (
            lambda value: value["dimensions"].update({"severity": "high"}),
            "unsupported field",
        ),
        (
            lambda value: value["dimensions"].update({"impact": "  "}),
            "non-empty string",
        ),
        (
            lambda value: value.update({"suggested_focus": []}),
            "between 1 and",
        ),
        (
            lambda value: value.update({"reasons": [""]}),
            "non-empty string",
        ),
        (
            lambda value: value.update({"reasons": [" padded "]}),
            "leading or trailing whitespace",
        ),
    ],
)
def test_parse_risk_proposal_rejects_shape_empty_text_and_counts(mutate, error):
    payload = _proposal_payload()
    mutate(payload)

    with pytest.raises(RiskProposalParseError, match=error):
        parse_risk_proposal(
            json.dumps(payload),
            signal_catalog=_packet().signal_catalog,
        )


def test_parse_risk_proposal_rejects_excessive_list_quantity():
    payload = _proposal_payload()
    payload["reasons"] = [f"reason {index}" for index in range(MAX_RISK_REASONS + 1)]

    with pytest.raises(RiskProposalParseError, match="between 1 and"):
        parse_risk_proposal(
            json.dumps(payload),
            signal_catalog=_packet().signal_catalog,
        )


def test_parse_risk_proposal_rejects_unknown_signal_ref():
    payload = _proposal_payload(signal_refs=["observation:not-authorized"])

    with pytest.raises(RiskProposalParseError, match="unauthorized ref"):
        parse_risk_proposal(
            json.dumps(payload),
            signal_catalog=_packet().signal_catalog,
        )


def test_parse_risk_proposal_rejects_duplicate_fields_and_non_standard_json():
    encoded = json.dumps(_proposal_payload())
    duplicate = encoded.replace(
        '"level": "high"',
        '"level": "high", "level": "low"',
        1,
    )
    non_standard = _proposal_payload()
    non_standard["dimensions"]["impact"] = float("nan")

    with pytest.raises(RiskProposalParseError, match="duplicate field: level"):
        parse_risk_proposal(duplicate, _packet())
    with pytest.raises(RiskProposalParseError, match="non-standard JSON constant"):
        parse_risk_proposal(json.dumps(non_standard), _packet())


def test_compile_risk_proposal_applies_floor_and_preserves_local_context():
    local = _local(RiskLevel.HIGH)
    proposal = RiskProposal(
        level=RiskLevel.LOW,
        dimensions=_dimensions(),
        reasons=["model reason", "local reason"],
        signal_refs=["changed_file:app.py"],
        uncertainties=["model uncertainty"],
        suggested_focus=["model focus"],
    )

    compiled = compile_risk_proposal(local, proposal)

    assert compiled.final_level is RiskLevel.HIGH
    assert compiled.floor_applied is True
    assert compiled.assessment.dimensions == _dimensions()
    assert compiled.assessment.reasons == ["local reason", "model reason"]
    assert compiled.assessment.signal_refs == [
        "quality_gate:python_compile",
        "changed_file:app.py",
    ]
    assert compiled.assessment.uncertainties == [
        "local uncertainty",
        "model uncertainty",
    ]
    assert compiled.assessment.suggested_focus == ["local focus", "model focus"]


def test_compile_risk_proposal_can_raise_risk_and_null_proposal_is_exact_fallback():
    local = _local(RiskLevel.LOW)
    proposal = parse_risk_proposal(
        json.dumps(_proposal_payload(level="critical")),
        signal_catalog=_packet().signal_catalog,
    )

    raised = compile_risk_proposal(local, proposal)
    fallback = compile_risk_proposal(local, None)

    assert raised.final_level is RiskLevel.CRITICAL
    assert raised.floor_applied is False
    assert fallback.assessment is local
    assert fallback.model_proposed_level is None


def test_model_runner_retries_parse_failure_with_one_turn_requests():
    invalid = _proposal_payload()
    invalid["extra"] = "forbidden"
    adapter = FakeToolCallingAdapter(
        script=[_final(invalid, raw_id=1), _final(_proposal_payload(), raw_id=2)]
    )

    run = run_model_risk_assessment(
        adapter,
        _packet(),
        _local(RiskLevel.LOW),
        review_id="review-123",
        model="risk-model",
        max_provider_attempts=2,
        max_elapsed_seconds=5,
    )

    assert run.status == "accepted"
    assert run.assessment.level is RiskLevel.HIGH
    assert run.decision.attempts == 2
    assert run.raw_response is not None
    assert run.raw_response.accepted_attempt == 2
    assert [item.response_kind for item in run.raw_response.attempts] == [
        "final",
        "final",
    ]
    assert len(adapter.requests) == 2
    assert all(request.tools == [] for request in adapter.requests)
    assert all(request.tool_results == [] for request in adapter.requests)
    assert (
        adapter.requests[0].parameters["invocation_id"]
        == adapter.requests[1].parameters["invocation_id"]
        == run.decision.invocation_id
    )
    assert adapter.requests[0].parameters["attempt"] == 1
    assert adapter.requests[1].parameters["attempt"] == 2
    assert "Runtime rejected" in adapter.requests[1].messages[-1]["content"]
    json.dumps(risk_model_run_to_dict(run))


def test_model_runner_falls_back_after_provider_and_allowlist_failures():
    def unavailable(_request):
        raise TimeoutError("provider unavailable")

    adapter = FakeToolCallingAdapter(
        script=[
            unavailable,
            _final(
                _proposal_payload(signal_refs=["observation:not-authorized"]),
                raw_id=2,
            ),
        ]
    )
    local = _local(RiskLevel.HIGH)

    run = run_model_risk_assessment(
        adapter,
        _packet(),
        local,
        review_id="review-123",
        model="risk-model",
        max_provider_attempts=2,
        max_elapsed_seconds=5,
    )

    assert run.status == "fallback"
    assert run.assessment is local
    assert run.proposal is None
    assert run.decision.fallback_used is True
    assert run.decision.local_floor is RiskLevel.HIGH
    assert run.decision.final_level is RiskLevel.HIGH
    assert "provider invocation failed" in run.decision.failure_reason
    assert "unauthorized ref" in run.decision.failure_reason
    assert len(adapter.requests) == 2
    json.dumps(run.to_dict())


def test_model_runner_rejects_response_returned_after_elapsed_budget():
    times = iter([0.0, 0.0, 1.0])
    adapter = FakeToolCallingAdapter(script=[_final(_proposal_payload())])
    local = _local(RiskLevel.MEDIUM)

    run = run_model_risk_assessment(
        adapter,
        _packet(),
        local,
        review_id="review-timeout",
        model="risk-model",
        max_provider_attempts=1,
        max_elapsed_seconds=1,
        clock=lambda: next(times),
    )

    assert run.status == "fallback"
    assert run.assessment is local
    assert run.raw_response is not None
    assert run.raw_response.attempts[0].response_kind == "timeout"
    assert "elapsed-time budget exhausted" in run.decision.failure_reason


def test_disabled_local_run_has_decision_but_no_fabricated_model_artifacts():
    local = _local(RiskLevel.MEDIUM)

    run = build_local_risk_run(
        _packet(),
        local,
        review_id="review-local",
    )

    assert run.status == "disabled"
    assert run.assessment is local
    assert run.envelope is None
    assert run.raw_response is None
    assert run.decision.model_status == "disabled"
    assert run.decision.fallback_used is False
    json.dumps(run.to_dict())


def test_input_digest_and_invocation_id_are_stable_and_scope_review_identity():
    first = _packet(
        quality_gates={"python_compile": "passed", "mypy": "unavailable"}
    )
    second = _packet(
        quality_gates={"mypy": "unavailable", "python_compile": "passed"}
    )

    first_digest = risk_input_digest(first)
    second_digest = risk_input_digest(second)

    assert first_digest == second_digest
    assert len(first_digest) == 64
    assert risk_invocation_id("review-1", first_digest) == risk_invocation_id(
        "review-1",
        second_digest,
    )
    assert risk_invocation_id("review-1", first_digest) != risk_invocation_id(
        "review-2",
        first_digest,
    )


def test_model_risk_compiler_consumes_typed_memory_floor_not_memory_statement() -> None:
    memory_id = "MEM-" + "5" * 64
    reference = MemoryReference(
        memory_id=memory_id,
        kind="high_risk_module",
        source_refs=("memory-source:" + "6" * 64,),
    )
    projection = RiskMemoryProjection(
        signals=(
            MemoryRiskSignal(
                signal_ref=f"memory:{memory_id}",
                summary="Prior incidents make this module sensitive.",
                memory=reference,
            ),
        ),
        risk_floor=CompiledRiskFloor(
            minimum_level=RiskLevel.HIGH,
            memory_ids=(memory_id,),
        ),
    )
    proposal = RiskProposal(
        level=RiskLevel.LOW,
        dimensions=_dimensions(),
        reasons=["model considers the change local"],
        signal_refs=[f"memory:{memory_id}"],
        uncertainties=["model uncertainty"],
        suggested_focus=["model focus"],
    )

    compiled = compile_risk_proposal(
        _local(RiskLevel.LOW),
        proposal,
        memory_projection=projection,
    )
    model_input = risk_packet_to_model_input(
        replace(_packet(), memory_projection=projection)
    )

    assert compiled.local_floor is RiskLevel.HIGH
    assert compiled.final_level is RiskLevel.HIGH
    assert compiled.floor_applied is True
    assert "memory" not in model_input
    encoded = json.dumps(model_input)
    assert memory_id not in encoded
    assert "Prior incidents make this module sensitive" not in encoded
    assert "risk_floor" not in encoded
    assert "eligible_records" not in encoded
    assert "policy_effect" not in encoded


def test_informational_memory_is_not_a_model_authority_channel() -> None:
    memory_id = "MEM-" + "7" * 64
    statement = "A prior incident should not silently become a risk floor."
    reference = MemoryReference(
        memory_id=memory_id,
        kind="incident_lesson",
        source_refs=("memory-source:" + "8" * 64,),
        local_only=True,
    )
    projection = RiskMemoryProjection(
        signals=(
            MemoryRiskSignal(
                signal_ref=f"memory:{memory_id}",
                summary=statement,
                memory=reference,
            ),
        ),
    )
    packet = replace(
        _packet(),
        signal_catalog={
            **_packet().signal_catalog,
            f"memory:{memory_id}": statement,
        },
        memory_projection=projection,
    )

    compiled = compile_risk_proposal(
        _local(RiskLevel.LOW),
        None,
        memory_projection=projection,
    )
    encoded = json.dumps(risk_packet_to_model_input(packet))

    assert compiled.final_level is RiskLevel.LOW
    assert memory_id not in encoded
    assert statement not in encoded
    assert "local_only" not in encoded
