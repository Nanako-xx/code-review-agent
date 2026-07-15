from dataclasses import replace

from review_agent.completion import check_completion, completion_to_dict
from review_agent.evidence import ContractCoverage, EvidenceReconciliation
from review_agent.model_protocol import ModelResponse
from review_agent.models import (
    CompiledMemoryRequirement,
    CompletionMemoryProjection,
    IntentPacket,
    IntentSource,
    IntentStatus,
    MemoryDiagnostic,
    MemoryDiagnosticCode,
    QualityGateResult,
    ReviewerResult,
    ReviewerResultStatus,
)
from review_agent.orchestrator import ReviewerExecution
from review_agent.quality import QualityGateDefinition, QualityGatePlan
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


def test_completion_prefers_new_core_role_kind_over_role_name():
    misleading = execution(0, "Core Reviewer", ReviewerResultStatus.FAILED)
    misleading = replace(
        misleading,
        assignment=replace(
            misleading.assignment,
            role_kind="specialist",
            perspective_key="security",
            planner_source="model",
        ),
    )

    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[misleading],
        reconciliation=reconciliation(),
    )

    assert "Core Reviewer did not run" in result.blockers
    assert "Core Reviewer failed" not in result.blockers
    assert result.missing_perspectives == ["Core Reviewer"]


def test_completion_accepts_new_core_kind_without_legacy_core_name():
    typed_core = execution(0, "Primary Reviewer", ReviewerResultStatus.FAILED)
    typed_core = replace(
        typed_core,
        assignment=replace(
            typed_core.assignment,
            role_kind="core",
            perspective_key="core",
            planner_source="runtime_injected",
        ),
    )

    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[typed_core],
        reconciliation=reconciliation(),
    )

    assert "Core Reviewer did not run" not in result.blockers
    assert "Primary Reviewer failed" in result.blockers


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


def planned_gate(*, blocking=False):
    return QualityGateDefinition(
        name="pytest",
        category="test",
        cost="expensive",
        source="builtin",
        command=["python", "-m", "pytest", "-q"],
        blocking=blocking,
        trigger_risks=["medium", "high", "critical"],
    )


def gate_result(gate, status, *, reason=None):
    return QualityGateResult(
        name=gate.name,
        status=status,
        command=list(gate.command),
        summary=f"pytest {status}",
        observation_ref="O-quality-pytest",
        category=gate.category,
        cost=gate.cost,
        source=gate.source,
        blocking=gate.blocking,
        reason=reason,
        sandbox="test-sandbox",
    )


def completion_with_gate(gate, results, *, issues=None, known_observations=None):
    return check_completion(
        intent=intent(),
        quality_results=results,
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)],
        reconciliation=reconciliation_with_coverage(coverage(0, "Core Reviewer")),
        quality_plan=QualityGatePlan(
            revision="a" * 40,
            gates=[gate],
            discovery_issues=list(issues or []),
        ),
        quality_observation_refs=known_observations,
    )


def test_completion_requires_every_planned_quality_gate_result():
    result = completion_with_gate(planned_gate(), [])

    assert result.status == "blocked"
    assert "Quality gate result missing: pytest" in result.blockers


def test_completion_rejects_unknown_quality_gate_observation_ref():
    gate = planned_gate()
    result = completion_with_gate(
        gate,
        [gate_result(gate, "passed")],
        known_observations=set(),
    )

    assert result.status == "blocked"
    assert "Quality gate observation unknown: pytest" in result.blockers


def test_non_blocking_quality_failure_is_an_uncertainty_not_a_hard_gate():
    gate = planned_gate()
    result = completion_with_gate(gate, [gate_result(gate, "failed")])

    assert result.status == "completed_with_uncertainties"
    assert result.blockers == []
    assert any("Quality gate failed: pytest" in item for item in result.uncertainties)


def test_policy_skipped_non_blocking_gate_satisfies_planned_depth():
    gate = planned_gate()
    result = completion_with_gate(
        gate,
        [gate_result(gate, "skipped", reason="low risk policy")],
    )

    assert result.status == "completed"
    assert result.recommendation == "approve"


def test_repository_blocking_quality_gate_blocks_completion():
    gate = planned_gate(blocking=True)
    result = completion_with_gate(
        gate,
        [gate_result(gate, "unavailable", reason="pytest is not installed")],
    )

    assert result.status == "blocked"
    assert any("Quality gate unavailable: pytest" in item for item in result.blockers)


def test_quality_discovery_issue_is_preserved_as_manual_review_uncertainty():
    gate = planned_gate()
    result = completion_with_gate(
        gate,
        [gate_result(gate, "passed")],
        issues=["invalid explicit gate"],
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert "Quality gate discovery issue: invalid explicit gate" in result.uncertainties


def _completion_with_semantic(semantic, *, intent_status=IntentStatus.SUFFICIENT):
    return check_completion(
        intent=intent(intent_status),
        quality_results=[],
        executions=[
            execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)
        ],
        reconciliation=reconciliation_with_coverage(
            coverage(0, "Core Reviewer")
        ),
        semantic_reconciliation=semantic,
    )


def test_semantic_fallback_and_remaining_disagreement_require_manual_review():
    result = _completion_with_semantic(
        {
            "status": "fallback",
            "model": {"status": "fallback"},
            "remaining_disagreements": [{"issue": "behavior remains unknown"}],
            "supplemental": {"status": "not_needed", "stop_reason": "no_requests"},
        }
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert "Semantic reconciliation used deterministic fallback" in result.uncertainties
    assert "reviewer disagreements remain unresolved" in result.uncertainties


def test_supplemental_unavailable_requires_manual_review():
    result = _completion_with_semantic(
        {
            "status": "partial",
            "model": {"status": "accepted"},
            "remaining_disagreements": [],
            "supplemental": {"status": "unavailable", "stop_reason": "unavailable"},
        }
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert "Supplemental investigation unavailable" in result.uncertainties


def test_max_waves_maps_to_budget_exhausted_and_blockers_still_take_priority():
    semantic = {
        "status": "partial",
        "model": {"status": "accepted"},
        "remaining_disagreements": [{"issue": "still open"}],
        "supplemental": {
            "status": "budget_exhausted",
            "stop_reason": "max_waves",
        },
    }
    exhausted = _completion_with_semantic(semantic)
    blocked = _completion_with_semantic(
        semantic,
        intent_status=IntentStatus.INSUFFICIENT,
    )

    assert exhausted.status == "budget_exhausted"
    assert exhausted.recommendation == "manual_review"
    assert blocked.status == "blocked"
    assert blocked.recommendation == "manual_review"
    assert "Intent Packet insufficient" in blocked.blockers


def test_supplemental_execution_cannot_supply_initial_core_presence():
    supplemental = execution(
        0,
        "Supplemental Core Reviewer",
        ReviewerResultStatus.COMPLETED,
    )
    supplemental = replace(
        supplemental,
        assignment=replace(
            supplemental.assignment,
            role_kind="core",
            planner_source="semantic_reconciler",
        ),
    )

    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[supplemental],
        reconciliation=reconciliation(),
    )

    assert result.status == "blocked"
    assert "Core Reviewer did not run" in result.blockers


def test_completion_enforces_compiled_memory_contract_and_check_without_statements() -> None:
    memory_id = "MEM-" + "b" * 64
    projection = CompletionMemoryProjection(
        required_contracts=(
            CompiledMemoryRequirement(
                requirement_id="regression_safety",
                memory_ids=(memory_id,),
            ),
        ),
        required_checks=(
            CompiledMemoryRequirement(
                requirement_id="schema_check",
                memory_ids=(memory_id,),
            ),
        ),
    )
    kwargs = {
        "intent": intent(),
        "executions": [
            execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)
        ],
        "reconciliation": reconciliation_with_coverage(
            coverage(0, "Core Reviewer")
        ),
        "memory_projection": projection,
    }

    passed = check_completion(
        quality_results=[
            QualityGateResult(
                name="schema_check",
                status="passed",
                command=["python", "-m", "schema_check"],
                summary="schema valid",
            )
        ],
        **kwargs,
    )
    missing = check_completion(quality_results=[], **kwargs)

    assert passed.status == "completed"
    assert missing.status == "blocked"
    assert "Memory-required check result missing: schema_check" in missing.blockers
    assert "statement" not in completion_to_dict(passed)


def test_memory_required_contract_is_distributed_but_all_assigned_copies_must_complete() -> None:
    memory_id = "MEM-" + "e" * 64
    projection = CompletionMemoryProjection(
        required_contracts=(
            CompiledMemoryRequirement("api_compatibility", (memory_id,)),
        ),
    )
    core = execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)
    core = replace(
        core,
        assignment=replace(
            core.assignment,
            assigned_contract=["regression_safety", "api_compatibility"],
        ),
    )
    specialist = execution(
        1,
        "Compatibility Reviewer",
        ReviewerResultStatus.COMPLETED,
    )
    specialist = replace(
        specialist,
        assignment=replace(
            specialist.assignment,
            assigned_contract=["regression_safety"],
        ),
    )
    rows = [
        coverage(0, "Core Reviewer"),
        coverage(0, "Core Reviewer", "api_compatibility"),
        coverage(1, "Compatibility Reviewer"),
    ]

    distributed = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[core, specialist],
        reconciliation=reconciliation_with_coverage(*rows),
        memory_projection=projection,
    )
    duplicated_assignment = replace(
        specialist,
        assignment=replace(
            specialist.assignment,
            assigned_contract=["regression_safety", "api_compatibility"],
        ),
    )
    incomplete = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[core, duplicated_assignment],
        reconciliation=reconciliation_with_coverage(*rows),
        memory_projection=projection,
    )

    assert distributed.status == "completed"
    assert incomplete.status == "blocked"
    assert any(
        "Memory-required contract coverage incomplete: api_compatibility" in item
        for item in incomplete.blockers
    )


def test_memory_unavailable_stale_and_hard_overflow_are_visible_and_fail_closed() -> None:
    base = {
        "intent": intent(),
        "quality_results": [],
        "executions": [
            execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)
        ],
        "reconciliation": reconciliation_with_coverage(
            coverage(0, "Core Reviewer")
        ),
    }
    unavailable = MemoryDiagnostic(
        code=MemoryDiagnosticCode.UNAVAILABLE,
        message="Memory Store could not be opened",
    )
    stale = MemoryDiagnostic(
        code=MemoryDiagnosticCode.STALE,
        message="one approved record requires revalidation",
        memory_ids=("MEM-" + "c" * 64,),
    )
    hard_overflow = MemoryDiagnostic(
        code=MemoryDiagnosticCode.HARD_POLICY_OVERFLOW,
        message="typed hard policy exceeded the projection budget",
        memory_ids=("MEM-" + "d" * 64,),
    )

    degraded = check_completion(
        **base,
        memory_projection=CompletionMemoryProjection(
            diagnostics=(unavailable, stale),
        ),
    )
    blocked = check_completion(
        **base,
        memory_projection=CompletionMemoryProjection(
            diagnostics=(hard_overflow,),
        ),
    )

    assert degraded.status == "completed_with_uncertainties"
    assert degraded.recommendation == "manual_review"
    assert any("memory_unavailable" in item for item in degraded.uncertainties)
    assert any("stale" in item for item in degraded.uncertainties)
    assert blocked.status == "blocked"
    assert any("hard_policy_overflow" in item for item in blocked.blockers)
    assert "memory_diagnostics" in completion_to_dict(blocked)


def test_legacy_completion_payload_has_no_memory_fields() -> None:
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)],
        reconciliation=reconciliation_with_coverage(coverage(0, "Core Reviewer")),
    )

    assert "memory_diagnostics" not in completion_to_dict(result)
