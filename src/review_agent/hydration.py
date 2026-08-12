from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any, TypeVar

from review_agent.brief import (
    BriefFinding,
    RejectedHypothesis,
    ReviewBrief,
    build_memory_audit_projection,
)
from review_agent.completion import CompletionResult
from review_agent.evidence import (
    CanonicalFinding,
    ContractCoverage,
    EvidenceReconciliation,
    RejectedFinding,
)
from review_agent.final_risk import FinalRiskAssessment, final_risk_to_dict
from review_agent.incremental import (
    incremental_priority_from_dict,
    incremental_priority_to_dict,
)
from review_agent.memory_models import (
    FeedbackCalibrationSummary,
    MemorySelectionDecision,
    MemorySelectionInput,
    MemorySnapshot,
)
from review_agent.model_protocol import ModelResponse
from review_agent.models import (
    Assignment,
    ClarificationQuestion,
    ClarificationStatus,
    CompiledRiskFloor,
    ConclusionImpact,
    ContractAssessment,
    ContractItemStatus,
    DEFAULT_REVIEWER_MAX_ELAPSED_SECONDS,
    DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS,
    DEFAULT_REVIEWER_MAX_PROVIDER_ATTEMPTS,
    DEFAULT_REVIEWER_MAX_TOTAL_TOKENS,
    InitialContext,
    IntentClaim,
    IntentClaimState,
    IntentConfidence,
    IntentDecision,
    IntentDecisionAction,
    IntentField,
    IntentOrigin,
    IntentPacket,
    IntentSource,
    IntentStatus,
    MemoryDiagnostic,
    MemoryDiagnosticCode,
    MemoryReference,
    MemoryRiskSignal,
    ModelInvocationEnvelope,
    QualityGateResult,
    ReviewRequest,
    ReviewerFinding,
    ReviewerResult,
    ReviewerResultStatus,
    ReviewerRuntimeMetadata,
    ReviewerTerminationReason,
    RiskAssessment,
    RiskAssessmentPacket,
    RiskMemoryProjection,
    RiskLevel,
)
from review_agent.orchestrator import ReviewerExecution
from review_agent.quality import QualityGateDefinition, QualityGatePlan
from review_agent.reconciler import (
    SemanticReconciliation,
    semantic_reconciliation_from_dict as _semantic_reconciliation_from_dict,
)
from review_agent.reviewer_runtime import reviewer_runtime_to_dict
from review_agent.reviewer_executor import (
    ReviewerExecutionResultV2,
    reviewer_execution_result_v2_from_dict as _reviewer_execution_result_v2_from_dict,
)
from review_agent.repository_intelligence import (
    ChangedSymbol,
    RepositoryIntelligenceSnapshot,
)
from review_agent.review_protocol import ReviewerOutput as ReviewerOutputV2
from review_agent.reviewer_output import RejectedReviewerFinding


def reviewer_output_v2_from_dict(
    payload: Mapping[str, Any],
) -> ReviewerOutputV2:
    """Hydrate a persisted, already validated ReviewerOutput v2."""

    return ReviewerOutputV2.from_dict(payload)


def rejected_reviewer_finding_v2_from_dict(
    payload: Mapping[str, Any],
) -> RejectedReviewerFinding:
    """Hydrate one stable candidate-level rejection record."""

    return RejectedReviewerFinding.from_dict(payload)


def reviewer_execution_result_v2_from_dict(
    payload: Mapping[str, Any],
) -> ReviewerExecutionResultV2:
    """Hydrate one persisted terminal v6 Reviewer execution result."""

    return _reviewer_execution_result_v2_from_dict(payload)


def semantic_reconciliation_from_dict(
    payload: Mapping[str, Any],
) -> SemanticReconciliation:
    """Hydrate the authoritative semantic reconciliation artifact strictly."""

    return _semantic_reconciliation_from_dict(payload)


def memory_selection_input_from_dict(
    payload: Mapping[str, Any],
) -> MemorySelectionInput:
    """Hydrate a Memory Selection input through its strict model boundary."""

    return MemorySelectionInput.from_dict(payload)


def memory_snapshot_from_dict(payload: Mapping[str, Any]) -> MemorySnapshot:
    """Hydrate a pinned Memory Snapshot through its strict model boundary."""

    return MemorySnapshot.from_dict(payload)


def memory_selection_decision_from_dict(
    payload: Mapping[str, Any],
) -> MemorySelectionDecision:
    """Hydrate a Memory Selection decision through its strict model boundary."""

    return MemorySelectionDecision.from_dict(payload)


def feedback_calibration_summary_from_dict(
    payload: Mapping[str, Any],
) -> FeedbackCalibrationSummary:
    """Hydrate feedback calibration through its strict model boundary."""

    return FeedbackCalibrationSummary.from_dict(payload)


def review_request_from_dict(payload: Mapping[str, Any]) -> ReviewRequest:
    item = _object(payload, "review_request")
    _exact(
        item,
        {
            "repository_path",
            "base_revision",
            "head_revision",
            "title",
            "description",
            "linked_requirements",
            "user_intent",
            "review_focus",
            "project_rules",
            "existing_ci_evidence",
        },
        "review_request",
    )
    return ReviewRequest(
        repository_path=_string(item, "repository_path", "review_request"),
        base_revision=_string(item, "base_revision", "review_request"),
        head_revision=_string(item, "head_revision", "review_request"),
        title=_optional_string(item, "title", "review_request"),
        description=_optional_string(item, "description", "review_request"),
        linked_requirements=tuple(
            _string_list(item, "linked_requirements", "review_request")
        ),
        user_intent=_optional_string(item, "user_intent", "review_request"),
        review_focus=_optional_string(item, "review_focus", "review_request"),
        project_rules=tuple(_string_list(item, "project_rules", "review_request")),
        existing_ci_evidence=tuple(
            _string_list(item, "existing_ci_evidence", "review_request")
        ),
    )


def intent_from_dict(payload: Mapping[str, Any]) -> IntentPacket:
    item = _object(payload, "intent")
    _required_with_optional(
        item,
        {
            "goal",
            "acceptance_criteria",
            "scope",
            "constraints",
            "sources",
            "status",
            "uncertainties",
        },
        {"provenance", "clarifications"},
        "intent",
    )
    sources_payload = _object_field(item, "sources", "intent")
    sources: dict[str, IntentSource] = {}
    for key, value in sources_payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError("intent.sources keys must be non-empty strings")
        sources[key] = _enum(IntentSource, value, f"intent.sources.{key}")
    return IntentPacket(
        goal=_optional_string(item, "goal", "intent"),
        acceptance_criteria=_string_list(item, "acceptance_criteria", "intent"),
        scope=_string_list(item, "scope", "intent"),
        constraints=_string_list(item, "constraints", "intent"),
        sources=sources,
        status=_enum_field(IntentStatus, item, "status", "intent"),
        uncertainties=_string_list(item, "uncertainties", "intent"),
        provenance=(
            [_intent_claim(row, f"intent.provenance[{index}]") for index, row in enumerate(_list_field(item, "provenance", "intent"))]
            if "provenance" in item
            else []
        ),
        clarifications=(
            [
                _clarification_question(
                    row,
                    f"intent.clarifications[{index}]",
                )
                for index, row in enumerate(
                    _list_field(item, "clarifications", "intent")
                )
            ]
            if "clarifications" in item
            else []
        ),
    )


def intent_claims_from_dict(payload: Mapping[str, Any]) -> list[IntentClaim]:
    item = _object(payload, "intent_candidates")
    _required_with_optional(
        item,
        {"claims"},
        {"uncertainties"},
        "intent_candidates",
    )
    return [
        _intent_claim(row, f"intent_candidates.claims[{index}]")
        for index, row in enumerate(
            _list_field(item, "claims", "intent_candidates")
        )
    ]


def clarification_questions_from_dict(
    payload: Mapping[str, Any],
) -> list[ClarificationQuestion]:
    item = _object(payload, "intent_questions")
    _exact(item, {"questions"}, "intent_questions")
    return [
        _clarification_question(row, f"intent_questions.questions[{index}]")
        for index, row in enumerate(
            _list_field(item, "questions", "intent_questions")
        )
    ]


def intent_decision_from_dict(payload: Mapping[str, Any]) -> IntentDecision:
    item = _object(payload, "intent_decision")
    _exact(
        item,
        {
            "question_id",
            "action",
            "corrected_values",
            "user_response",
            "continuation_basis",
            "decision_id",
        },
        "intent_decision",
    )
    return IntentDecision(
        question_id=_string(item, "question_id", "intent_decision"),
        action=_enum_field(
            IntentDecisionAction,
            item,
            "action",
            "intent_decision",
        ),
        corrected_values=_string_list(
            item,
            "corrected_values",
            "intent_decision",
        ),
        user_response=_optional_string(
            item,
            "user_response",
            "intent_decision",
        ),
        continuation_basis=_optional_string(
            item,
            "continuation_basis",
            "intent_decision",
        ),
        decision_id=_string(item, "decision_id", "intent_decision"),
    )


def _intent_claim(value: Any, context: str) -> IntentClaim:
    item = _object(value, context)
    _exact(
        item,
        {
            "field",
            "value",
            "source",
            "origin",
            "confidence",
            "source_refs",
            "evidence_refs",
            "claim_state",
            "conclusion_impact",
            "claim_id",
        },
        context,
    )
    return IntentClaim(
        field=_enum_field(IntentField, item, "field", context),
        value=_string(item, "value", context),
        source=_enum_field(IntentSource, item, "source", context),
        origin=_enum_field(IntentOrigin, item, "origin", context),
        confidence=_enum_field(IntentConfidence, item, "confidence", context),
        source_refs=_string_list(item, "source_refs", context),
        evidence_refs=_string_list(item, "evidence_refs", context),
        claim_state=_enum_field(IntentClaimState, item, "claim_state", context),
        conclusion_impact=_enum_field(
            ConclusionImpact,
            item,
            "conclusion_impact",
            context,
        ),
        claim_id=_string(item, "claim_id", context),
    )


def _clarification_question(value: Any, context: str) -> ClarificationQuestion:
    item = _object(value, context)
    _exact(
        item,
        {
            "field",
            "question",
            "rationale",
            "proposed_values",
            "claim_ids",
            "status",
            "user_response",
            "continuation_basis",
            "resolved_values",
            "decision_id",
            "question_id",
        },
        context,
    )
    return ClarificationQuestion(
        field=_enum_field(IntentField, item, "field", context),
        question=_string(item, "question", context),
        rationale=_string(item, "rationale", context),
        proposed_values=_string_list(item, "proposed_values", context),
        claim_ids=_string_list(item, "claim_ids", context),
        status=_enum_field(ClarificationStatus, item, "status", context),
        user_response=_optional_string(item, "user_response", context),
        continuation_basis=_optional_string(
            item,
            "continuation_basis",
            context,
        ),
        resolved_values=_string_list(item, "resolved_values", context),
        decision_id=_optional_string(item, "decision_id", context),
        question_id=_string(item, "question_id", context),
    )


def risk_packet_from_dict(payload: Mapping[str, Any]) -> RiskAssessmentPacket:
    item = _object(payload, "risk_packet")
    _required_with_optional(
        item,
        {
            "change_summary",
            "deterministic_signals",
            "intent_status",
            "intent_uncertainties",
            "diff_excerpt",
        },
        {"changed_symbols", "signal_catalog", "memory_projection"},
        "risk_packet",
    )
    changed_symbols: list[dict[str, object]] = []
    if "changed_symbols" in item:
        changed_symbols = [
            dict(row)
            for row in _object_list(item, "changed_symbols", "risk_packet")
        ]
    return RiskAssessmentPacket(
        change_summary=dict(_object_field(item, "change_summary", "risk_packet")),
        deterministic_signals=dict(
            _object_field(item, "deterministic_signals", "risk_packet")
        ),
        intent_status=_enum_field(
            IntentStatus,
            item,
            "intent_status",
            "risk_packet",
        ),
        intent_uncertainties=_string_list(
            item,
            "intent_uncertainties",
            "risk_packet",
        ),
        diff_excerpt=_string_list(item, "diff_excerpt", "risk_packet"),
        changed_symbols=changed_symbols,
        signal_catalog=(
            _string_mapping(item, "signal_catalog", "risk_packet")
            if "signal_catalog" in item
            else {}
        ),
        memory_projection=(
            _risk_memory_projection(
                item["memory_projection"],
                "risk_packet.memory_projection",
            )
            if item.get("memory_projection") is not None
            else None
        ),
    )


def _risk_memory_projection(value: Any, context: str) -> RiskMemoryProjection:
    item = _object(value, context)
    _required_with_optional(
        item,
        {"signals", "risk_floor", "diagnostics"},
        {"policy_sources"},
        context,
    )

    signals: list[MemoryRiskSignal] = []
    for index, raw_signal in enumerate(_list_field(item, "signals", context)):
        signal_context = f"{context}.signals[{index}]"
        signal = _object(raw_signal, signal_context)
        _exact(signal, {"signal_ref", "summary", "memory"}, signal_context)
        signals.append(
            MemoryRiskSignal(
                signal_ref=_string(signal, "signal_ref", signal_context),
                summary=_string(signal, "summary", signal_context),
                memory=_memory_reference(
                    signal["memory"],
                    f"{signal_context}.memory",
                ),
            )
        )

    raw_floor = item["risk_floor"]
    risk_floor: CompiledRiskFloor | None = None
    if raw_floor is not None:
        floor_context = f"{context}.risk_floor"
        floor = _object(raw_floor, floor_context)
        _exact(floor, {"minimum_level", "memory_ids"}, floor_context)
        risk_floor = CompiledRiskFloor(
            minimum_level=_enum_field(
                RiskLevel,
                floor,
                "minimum_level",
                floor_context,
            ),
            memory_ids=tuple(_string_list(floor, "memory_ids", floor_context)),
        )

    if "policy_sources" in item and not isinstance(item["policy_sources"], list):
        raise ValueError(f"{context}.policy_sources must be a list")
    policy_sources = tuple(
        _memory_reference(raw, f"{context}.policy_sources[{index}]")
        for index, raw in enumerate(item.get("policy_sources", []))
    )
    diagnostics = tuple(
        _memory_diagnostic(raw, f"{context}.diagnostics[{index}]")
        for index, raw in enumerate(_list_field(item, "diagnostics", context))
    )

    return RiskMemoryProjection(
        signals=tuple(signals),
        risk_floor=risk_floor,
        policy_sources=policy_sources,
        diagnostics=diagnostics,
    )


def _memory_diagnostic(value: Any, context: str) -> MemoryDiagnostic:
    diagnostic = _object(value, context)
    _exact(
        diagnostic,
        {"code", "message", "memory_ids", "blocking"},
        context,
    )
    return MemoryDiagnostic(
        code=_enum_field(MemoryDiagnosticCode, diagnostic, "code", context),
        message=_string(diagnostic, "message", context),
        memory_ids=tuple(_string_list(diagnostic, "memory_ids", context)),
        blocking=_boolean(diagnostic, "blocking", context),
    )


def _memory_reference(value: Any, context: str) -> MemoryReference:
    item = _object(value, context)
    _exact(item, {"memory_id", "kind", "source_refs", "local_only"}, context)
    return MemoryReference(
        memory_id=_string(item, "memory_id", context),
        kind=_string(item, "kind", context),
        source_refs=tuple(_string_list(item, "source_refs", context)),
        local_only=_boolean(item, "local_only", context),
    )


def risk_assessment_from_dict(payload: Mapping[str, Any]) -> RiskAssessment:
    item = _object(payload, "risk_assessment")
    _exact(
        item,
        {
            "level",
            "dimensions",
            "reasons",
            "signal_refs",
            "uncertainties",
            "suggested_focus",
        },
        "risk_assessment",
    )
    return RiskAssessment(
        level=_enum_field(RiskLevel, item, "level", "risk_assessment"),
        dimensions=_string_mapping(item, "dimensions", "risk_assessment"),
        reasons=_string_list(item, "reasons", "risk_assessment"),
        signal_refs=_string_list(item, "signal_refs", "risk_assessment"),
        uncertainties=_string_list(item, "uncertainties", "risk_assessment"),
        suggested_focus=_string_list(item, "suggested_focus", "risk_assessment"),
    )


def assignments_from_dict(payload: Mapping[str, Any]) -> list[Assignment]:
    item = _object(payload, "assignments")
    _exact(item, {"assignments"}, "assignments")
    rows = _list_field(item, "assignments", "assignments")
    return [_assignment_from_dict(row, f"assignments.assignments[{index}]") for index, row in enumerate(rows)]


def quality_results_from_dict(payload: Mapping[str, Any]) -> list[QualityGateResult]:
    item = _object(payload, "quality_results")
    _exact(item, {"results"}, "quality_results")
    rows = _list_field(item, "results", "quality_results")
    results: list[QualityGateResult] = []
    for index, row in enumerate(rows):
        context = f"quality_results.results[{index}]"
        value = _object(row, context)
        results.append(_quality_gate_result_from_mapping(value, context))
    return results


def quality_gate_plan_from_dict(payload: Mapping[str, Any]) -> QualityGatePlan:
    item = _object(payload, "quality_gate_plan")
    _exact(
        item,
        {"revision", "gates", "discovery_issues"},
        "quality_gate_plan",
    )
    gates: list[QualityGateDefinition] = []
    for index, row in enumerate(_list_field(item, "gates", "quality_gate_plan")):
        context = f"quality_gate_plan.gates[{index}]"
        value = _object(row, context)
        _exact(
            value,
            {
                "name",
                "category",
                "cost",
                "source",
                "command",
                "blocking",
                "timeout_seconds",
                "trigger_risks",
            },
            context,
        )
        gates.append(
            QualityGateDefinition(
                name=_string(value, "name", context),
                category=_string(value, "category", context),
                cost=_string(value, "cost", context),
                source=_string(value, "source", context),
                command=_string_list(value, "command", context),
                blocking=_boolean(value, "blocking", context),
                timeout_seconds=_positive_number(
                    value,
                    "timeout_seconds",
                    context,
                ),
                trigger_risks=_string_list(value, "trigger_risks", context),
            )
        )
    return QualityGatePlan(
        revision=_string(item, "revision", "quality_gate_plan"),
        gates=gates,
        discovery_issues=_string_list(
            item,
            "discovery_issues",
            "quality_gate_plan",
        ),
    )


def _quality_gate_result_from_mapping(
    value: Mapping[str, Any],
    context: str,
) -> QualityGateResult:
    required = {"name", "status", "command", "summary", "observation_ref"}
    optional = {
        "category",
        "cost",
        "source",
        "blocking",
        "reason",
        "exit_code",
        "duration_seconds",
        "output_truncated",
        "sandbox",
    }
    _required_with_optional(value, required, optional, context)
    exit_code_value = value.get("exit_code")
    if exit_code_value is not None and type(exit_code_value) is not int:
        raise ValueError(f"{context}.exit_code must be an integer or null")
    return QualityGateResult(
        name=_string(value, "name", context),
        status=_string(value, "status", context),
        command=_string_list(value, "command", context),
        summary=_string(value, "summary", context),
        observation_ref=_optional_string(value, "observation_ref", context),
        category=(
            _string(value, "category", context)
            if "category" in value
            else "unknown"
        ),
        cost=_string(value, "cost", context) if "cost" in value else "cheap",
        source=(
            _string(value, "source", context)
            if "source" in value
            else "legacy"
        ),
        blocking=(
            _boolean(value, "blocking", context)
            if "blocking" in value
            else False
        ),
        reason=(
            _optional_string(value, "reason", context)
            if "reason" in value
            else None
        ),
        exit_code=exit_code_value,
        duration_seconds=(
            _non_negative_number(value, "duration_seconds", context)
            if "duration_seconds" in value
            else 0.0
        ),
        output_truncated=(
            _boolean(value, "output_truncated", context)
            if "output_truncated" in value
            else False
        ),
        sandbox=(
            _string(value, "sandbox", context)
            if "sandbox" in value
            else "legacy"
        ),
    )


def repository_intelligence_from_dict(
    payload: Mapping[str, Any],
) -> RepositoryIntelligenceSnapshot:
    item = _object(payload, "repository_intelligence")
    _exact(
        item,
        {
            "base_revision",
            "revision",
            "changed_symbols",
            "lsp_status",
            "fallback_strategy",
            "text_search_backend",
        },
        "repository_intelligence",
    )
    symbols: list[ChangedSymbol] = []
    for index, row in enumerate(
        _list_field(item, "changed_symbols", "repository_intelligence")
    ):
        context = f"repository_intelligence.changed_symbols[{index}]"
        value = _object(row, context)
        _exact(
            value,
            {"path", "qualified_name", "kind", "change_type", "line_start", "line_end"},
            context,
        )
        symbols.append(
            ChangedSymbol(
                path=_string(value, "path", context),
                qualified_name=_string(value, "qualified_name", context),
                kind=_string(value, "kind", context),
                change_type=_string(value, "change_type", context),
                line_start=_integer(value, "line_start", context),
                line_end=_integer(value, "line_end", context),
            )
        )
    return RepositoryIntelligenceSnapshot(
        base_revision=_string(item, "base_revision", "repository_intelligence"),
        revision=_string(item, "revision", "repository_intelligence"),
        changed_symbols=symbols,
        lsp_status=_string(item, "lsp_status", "repository_intelligence"),
        fallback_strategy=_string(item, "fallback_strategy", "repository_intelligence"),
        text_search_backend=_string(item, "text_search_backend", "repository_intelligence"),
    )


def reviewer_result_from_dict(payload: Mapping[str, Any]) -> ReviewerResult:
    item = _object(payload, "reviewer_result")
    _exact(
        item,
        {
            "contract_assessments",
            "confirmed_findings",
            "rejected_hypotheses",
            "uncertainties",
            "observation_refs",
            "investigation_summary",
            "status",
        },
        "reviewer_result",
    )
    assessments: list[ContractAssessment] = []
    for index, row in enumerate(
        _list_field(item, "contract_assessments", "reviewer_result")
    ):
        context = f"reviewer_result.contract_assessments[{index}]"
        value = _object(row, context)
        _exact(value, {"contract", "status", "summary", "evidence_refs"}, context)
        assessments.append(
            ContractAssessment(
                contract=_string(value, "contract", context),
                status=_enum_field(ContractItemStatus, value, "status", context),
                summary=_string(value, "summary", context),
                evidence_refs=_string_list(value, "evidence_refs", context),
            )
        )
    findings: list[ReviewerFinding] = []
    for index, row in enumerate(
        _list_field(item, "confirmed_findings", "reviewer_result")
    ):
        context = f"reviewer_result.confirmed_findings[{index}]"
        value = _object(row, context)
        _required_with_optional(
            value,
            {"claim", "severity", "confidence", "evidence_refs", "suggested_action"},
            {"path", "line", "impact", "verification_performed"},
            context,
        )
        line = value.get("line")
        if line is not None and (type(line) is not int or line < 1):
            raise ValueError(f"{context}.line must be a positive integer or null")
        findings.append(
            ReviewerFinding(
                claim=_string(value, "claim", context),
                severity=_string(value, "severity", context),
                confidence=_string(value, "confidence", context),
                evidence_refs=_string_list(value, "evidence_refs", context),
                suggested_action=_optional_string(value, "suggested_action", context),
                path=(
                    _optional_string(value, "path", context)
                    if "path" in value
                    else None
                ),
                line=line,
                impact=(
                    _string(value, "impact", context)
                    if "impact" in value
                    else ""
                ),
                verification_performed=(
                    _string_list(value, "verification_performed", context)
                    if "verification_performed" in value
                    else []
                ),
            )
        )
    return ReviewerResult(
        contract_assessments=assessments,
        confirmed_findings=findings,
        rejected_hypotheses=_string_list(item, "rejected_hypotheses", "reviewer_result"),
        uncertainties=_string_list(item, "uncertainties", "reviewer_result"),
        observation_refs=_string_list(item, "observation_refs", "reviewer_result"),
        investigation_summary=_string(item, "investigation_summary", "reviewer_result"),
        status=_enum_field(ReviewerResultStatus, item, "status", "reviewer_result"),
    )


def reviewer_execution_from_artifacts(
    *,
    reviewer_index: int,
    trace_id: str,
    assignment: Assignment,
    envelope_payload: Mapping[str, Any],
    response_payload: Mapping[str, Any],
    result_payload: Mapping[str, Any],
) -> ReviewerExecution:
    if type(reviewer_index) is not int or reviewer_index < 0:
        raise ValueError("reviewer_index must be a non-negative integer")
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError("trace_id must be a non-empty string")
    envelope = _model_envelope_from_dict(envelope_payload)
    response = _model_response_from_dict(response_payload)
    runtime = _reviewer_runtime_from_response(response_payload)
    return ReviewerExecution(
        reviewer_index=reviewer_index,
        trace_id=trace_id,
        assignment=assignment,
        envelope=envelope,
        response=response,
        result=reviewer_result_from_dict(result_payload),
        runtime=runtime,
    )


def reconciliation_from_dict(payload: Mapping[str, Any]) -> EvidenceReconciliation:
    item = _object(payload, "reconciliation")
    _exact(
        item,
        {
            "canonical_findings",
            "rejected_findings",
            "remaining_disagreements",
            "contract_coverage",
            "evidence_quality",
        },
        "reconciliation",
    )
    canonical = [
        _canonical_finding(row, f"reconciliation.canonical_findings[{index}]")
        for index, row in enumerate(
            _list_field(item, "canonical_findings", "reconciliation")
        )
    ]
    rejected = [
        _rejected_finding(row, f"reconciliation.rejected_findings[{index}]")
        for index, row in enumerate(
            _list_field(item, "rejected_findings", "reconciliation")
        )
    ]
    coverage = [
        _contract_coverage(row, f"reconciliation.contract_coverage[{index}]")
        for index, row in enumerate(
            _list_field(item, "contract_coverage", "reconciliation")
        )
    ]
    return EvidenceReconciliation(
        canonical_findings=canonical,
        rejected_findings=rejected,
        remaining_disagreements=_string_list(
            item,
            "remaining_disagreements",
            "reconciliation",
        ),
        contract_coverage=coverage,
        evidence_quality=_string(item, "evidence_quality", "reconciliation"),
    )


def completion_from_dict(payload: Mapping[str, Any]) -> CompletionResult:
    item = _object(payload, "completion")
    _required_with_optional(
        item,
        {"status", "recommendation", "blockers", "uncertainties", "missing_perspectives"},
        {"memory_diagnostics"},
        "completion",
    )
    diagnostics = tuple(
        _memory_diagnostic(raw, f"completion.memory_diagnostics[{index}]")
        for index, raw in enumerate(
            _list_field(item, "memory_diagnostics", "completion")
            if "memory_diagnostics" in item
            else []
        )
    )
    return CompletionResult(
        status=_string(item, "status", "completion"),
        recommendation=_string(item, "recommendation", "completion"),
        blockers=_string_list(item, "blockers", "completion"),
        uncertainties=_string_list(item, "uncertainties", "completion"),
        missing_perspectives=_string_list(item, "missing_perspectives", "completion"),
        memory_diagnostics=diagnostics,
    )


def final_risk_from_dict(payload: Mapping[str, Any]) -> FinalRiskAssessment:
    item = _object(payload, "final_risk")
    _required_with_optional(
        item,
        {
            "status",
            "initial_level",
            "level",
            "reasons",
            "escalations",
            "deescalations",
            "uncertainties",
            "signal_refs",
        },
        {"applied_memory", "memory_diagnostics", "residual_risk"},
        "final_risk",
    )
    memory_fields = {"applied_memory", "memory_diagnostics", "residual_risk"}
    present_memory_fields = memory_fields.intersection(item)
    if present_memory_fields and present_memory_fields != memory_fields:
        missing = ", ".join(sorted(memory_fields - present_memory_fields))
        raise ValueError(
            "final_risk is missing required Memory field(s): " + missing
        )
    applied_memory = tuple(
        _serialized_memory_reference(raw, f"final_risk.applied_memory[{index}]")
        for index, raw in enumerate(
            _list_field(item, "applied_memory", "final_risk")
            if "applied_memory" in item
            else []
        )
    )
    diagnostics = tuple(
        _memory_diagnostic(raw, f"final_risk.memory_diagnostics[{index}]")
        for index, raw in enumerate(
            _list_field(item, "memory_diagnostics", "final_risk")
            if "memory_diagnostics" in item
            else []
        )
    )
    return FinalRiskAssessment(
        status=_string(item, "status", "final_risk"),
        initial_level=_enum_field(RiskLevel, item, "initial_level", "final_risk"),
        level=_enum_field(RiskLevel, item, "level", "final_risk"),
        reasons=_string_list(item, "reasons", "final_risk"),
        escalations=_string_list(item, "escalations", "final_risk"),
        deescalations=_string_list(item, "deescalations", "final_risk"),
        uncertainties=_string_list(item, "uncertainties", "final_risk"),
        signal_refs=_string_list(item, "signal_refs", "final_risk"),
        applied_memory=applied_memory,
        memory_diagnostics=diagnostics,
        residual_risk=(
            tuple(_string_list(item, "residual_risk", "final_risk"))
            if "residual_risk" in item
            else ()
        ),
    )


def _serialized_memory_reference(value: Any, context: str) -> MemoryReference:
    item = _object(value, context)
    _exact(
        item,
        {"memory_id", "kind", "authority", "source_refs", "local_only"},
        context,
    )
    if _string(item, "authority", context) != "human_approved_project_memory":
        raise ValueError(f"{context}.authority is unsupported")
    return MemoryReference(
        memory_id=_string(item, "memory_id", context),
        kind=_string(item, "kind", context),
        source_refs=tuple(_string_list(item, "source_refs", context)),
        local_only=_boolean(item, "local_only", context),
    )


def review_brief_from_dict(payload: Mapping[str, Any]) -> ReviewBrief:
    item = _object(payload, "review_brief")
    _required_with_optional(
        item,
        {
            "review_id",
            "base_revision",
            "head_revision",
            "change_intent",
            "intent_assessment",
            "initial_and_final_risk_assessment",
            "quality_gates",
            "change_map_and_repository_impact",
            "verified_findings",
            "rejected_hypotheses",
            "uncertainties",
            "reviewer_disagreements",
            "review_contract_coverage",
            "verification_evidence",
            "human_review_checklist_and_reading_order",
            "non_binding_recommendation",
        },
        {"orchestration", "semantic_reconciliation", "memory_audit"},
        "review_brief",
    )
    findings = [
        _brief_finding(row, f"review_brief.verified_findings[{index}]")
        for index, row in enumerate(
            _list_field(item, "verified_findings", "review_brief")
        )
    ]
    rejected = [
        _rejected_hypothesis(row, f"review_brief.rejected_hypotheses[{index}]")
        for index, row in enumerate(
            _list_field(item, "rejected_hypotheses", "review_brief")
        )
    ]
    change_intent = _review_brief_change_intent(item["change_intent"])
    intent_assessment = _review_brief_intent_assessment(item["intent_assessment"])
    risk_assessment = _review_brief_risk_assessment(
        item["initial_and_final_risk_assessment"]
    )
    quality_gates = _review_brief_quality_gates(item["quality_gates"])
    change_map = _review_brief_change_map(item["change_map_and_repository_impact"])
    contract_coverage = _review_brief_contract_coverage(
        item["review_contract_coverage"]
    )
    verification_evidence = _review_brief_verification_evidence(
        item["verification_evidence"]
    )
    semantic_reconciliation: dict[str, Any] = {}
    if "semantic_reconciliation" in item:
        semantic_payload = dict(
            _object_field(item, "semantic_reconciliation", "review_brief")
        )
        if semantic_payload:
            semantic_reconciliation = _semantic_reconciliation_from_dict(
                semantic_payload
            ).to_dict()
    memory_audit = (
        _review_brief_memory_audit(item["memory_audit"])
        if "memory_audit" in item
        else {}
    )
    return ReviewBrief(
        review_id=_string(item, "review_id", "review_brief"),
        base_revision=_string(item, "base_revision", "review_brief"),
        head_revision=_string(item, "head_revision", "review_brief"),
        change_intent=change_intent,
        intent_assessment=intent_assessment,
        initial_and_final_risk_assessment=risk_assessment,
        quality_gates=quality_gates,
        change_map_and_repository_impact=change_map,
        verified_findings=findings,
        rejected_hypotheses=rejected,
        uncertainties=_string_list(item, "uncertainties", "review_brief"),
        reviewer_disagreements=_string_list(
            item,
            "reviewer_disagreements",
            "review_brief",
        ),
        review_contract_coverage=contract_coverage,
        verification_evidence=verification_evidence,
        human_review_checklist_and_reading_order=_string_list(
            item,
            "human_review_checklist_and_reading_order",
            "review_brief",
        ),
        non_binding_recommendation=_string(
            item,
            "non_binding_recommendation",
            "review_brief",
        ),
        orchestration=(
            dict(_object_field(item, "orchestration", "review_brief"))
            if "orchestration" in item
            else {}
        ),
        semantic_reconciliation=semantic_reconciliation,
        memory_audit=memory_audit,
    )


_AUDIT_MEMORY_ID = re.compile(r"^MEM-[0-9a-f]{64}$")
_AUDIT_CANDIDATE_ID = re.compile(r"^MC-[0-9a-f]{64}$")
_AUDIT_MEMORY_KINDS = {
    "architecture_boundary",
    "business_invariant",
    "review_rule",
    "compatibility_requirement",
    "verification_command",
    "incident_lesson",
    "high_risk_module",
    "unknown",
}


def _review_brief_memory_audit(value: Any) -> dict[str, Any]:
    context = "review_brief.memory_audit"
    item = dict(_object(value, context))
    if not item:
        return {}
    _exact(
        item,
        {
            "schema_version",
            "applied_memory",
            "not_applied_memory",
            "compiled_policy",
            "cache_provenance",
            "warnings",
            "feedback_summary",
            "pending_candidates",
            "status",
        }
        | ({"snapshot"} if "snapshot" in item else set()),
        context,
    )
    if _string(item, "schema_version", context) != "memory_audit_v1":
        raise ValueError(f"{context}.schema_version is unsupported")
    canonical = build_memory_audit_projection(item)
    if canonical != item:
        raise ValueError(f"{context} is not a canonical typed Memory audit projection")
    _validate_audit_json(item, context)

    for index, raw in enumerate(_list_field(item, "applied_memory", context)):
        row_context = f"{context}.applied_memory[{index}]"
        row = _object(raw, row_context)
        memory_id = _string(row, "memory_id", row_context)
        if _AUDIT_MEMORY_ID.fullmatch(memory_id) is None:
            raise ValueError(f"{row_context}.memory_id must be a stable Memory ID")
        if _string(row, "kind", row_context) not in _AUDIT_MEMORY_KINDS:
            raise ValueError(f"{row_context}.kind is unsupported")
        if _string(row, "authority", row_context) not in {
            "human_approved_context",
            "runtime_compiled_policy",
        }:
            raise ValueError(f"{row_context}.authority is unsupported")

    for index, raw in enumerate(_list_field(item, "not_applied_memory", context)):
        row_context = f"{context}.not_applied_memory[{index}]"
        row = _object(raw, row_context)
        memory_id = _string(row, "memory_id", row_context)
        if _AUDIT_MEMORY_ID.fullmatch(memory_id) is None:
            raise ValueError(f"{row_context}.memory_id must be a stable Memory ID")
        if _string(row, "authority", row_context) != "not_applied":
            raise ValueError(f"{row_context}.authority is unsupported")

    policy = _object_field(item, "compiled_policy", context)
    if policy:
        for collection in ("actions", "provenance"):
            for index, raw in enumerate(policy.get(collection, [])):
                row_context = f"{context}.compiled_policy.{collection}[{index}]"
                row = _object(raw, row_context)
                memory_ids = (
                    row.get("memory_ids", [])
                    if collection == "actions"
                    else [row.get("memory_id")]
                )
                if not isinstance(memory_ids, list):
                    raise ValueError(f"{row_context}.memory_ids must be a list")
                for memory_id in memory_ids:
                    if (
                        not isinstance(memory_id, str)
                        or _AUDIT_MEMORY_ID.fullmatch(memory_id) is None
                    ):
                        raise ValueError(
                            f"{row_context} contains an invalid Memory ID"
                        )

    for collection in ("warnings",):
        for index, raw in enumerate(_list_field(item, collection, context)):
            row = _object(raw, f"{context}.{collection}[{index}]")
            memory_id = row.get("memory_id")
            if memory_id is not None and (
                not isinstance(memory_id, str)
                or _AUDIT_MEMORY_ID.fullmatch(memory_id) is None
            ):
                raise ValueError(
                    f"{context}.{collection}[{index}].memory_id is invalid"
                )
    for index, raw in enumerate(_list_field(item, "pending_candidates", context)):
        row_context = f"{context}.pending_candidates[{index}]"
        row = _object(raw, row_context)
        candidate_id = _string(row, "candidate_id", row_context)
        if _AUDIT_CANDIDATE_ID.fullmatch(candidate_id) is None:
            raise ValueError(f"{row_context}.candidate_id is invalid")
        if row.get("active") is not False:
            raise ValueError(f"{row_context}.active must be false")
    return canonical


def _validate_audit_json(value: Any, context: str, *, depth: int = 0) -> None:
    if depth > 12:
        raise ValueError(f"{context} exceeds the maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > 8192:
            raise ValueError(f"{context} text exceeds the audit limit")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite number")
        return
    if isinstance(value, list):
        if len(value) > 256:
            raise ValueError(f"{context} exceeds the audit item limit")
        for index, item in enumerate(value):
            _validate_audit_json(item, f"{context}[{index}]", depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{context} keys must be non-empty strings")
            _validate_audit_json(item, f"{context}.{key}", depth=depth + 1)
        return
    raise ValueError(f"{context} contains an unsupported value")


def _review_brief_change_intent(value: Any) -> dict[str, Any]:
    context = "review_brief.change_intent"
    item = _object(value, context)
    _required_with_optional(
        item,
        {"goal", "acceptance_criteria", "scope", "constraints", "sources"},
        {"provenance"},
        context,
    )
    sources_payload = _object_field(item, "sources", context)
    sources: dict[str, str] = {}
    for key, source in sources_payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{context}.sources keys must be non-empty strings")
        sources[key] = _enum(IntentSource, source, f"{context}.sources.{key}").value
    result: dict[str, Any] = {
        "goal": _optional_string(item, "goal", context),
        "acceptance_criteria": _string_list(item, "acceptance_criteria", context),
        "scope": _string_list(item, "scope", context),
        "constraints": _string_list(item, "constraints", context),
        "sources": sources,
    }
    if "provenance" in item:
        result["provenance"] = [
            _review_brief_intent_claim(
                row,
                f"{context}.provenance[{index}]",
            )
            for index, row in enumerate(_list_field(item, "provenance", context))
        ]
    return result


def _review_brief_intent_assessment(value: Any) -> dict[str, Any]:
    context = "review_brief.intent_assessment"
    item = _object(value, context)
    _required_with_optional(
        item,
        {"status", "uncertainties", "source_counts"},
        {
            "clarification_history",
            "unresolved_questions",
            "unconfirmed_inferred_claims",
        },
        context,
    )
    result: dict[str, Any] = {
        "status": _enum_field(IntentStatus, item, "status", context).value,
        "uncertainties": _string_list(item, "uncertainties", context),
        "source_counts": _non_negative_integer_mapping(
            item,
            "source_counts",
            context,
        ),
    }
    if "clarification_history" in item:
        result["clarification_history"] = [
            _review_brief_clarification(
                row,
                f"{context}.clarification_history[{index}]",
            )
            for index, row in enumerate(
                _list_field(item, "clarification_history", context)
            )
        ]
    if "unresolved_questions" in item:
        unresolved_questions = [
            _review_brief_clarification(
                row,
                f"{context}.unresolved_questions[{index}]",
            )
            for index, row in enumerate(
                _list_field(item, "unresolved_questions", context)
            )
        ]
        if any(
            row["status"]
            not in {ClarificationStatus.PENDING.value, ClarificationStatus.OPEN.value}
            for row in unresolved_questions
        ):
            raise ValueError(
                f"{context}.unresolved_questions must contain only pending or open questions"
            )
        result["unresolved_questions"] = unresolved_questions
    if "unconfirmed_inferred_claims" in item:
        unconfirmed_claims = [
            _review_brief_intent_claim(
                row,
                f"{context}.unconfirmed_inferred_claims[{index}]",
            )
            for index, row in enumerate(
                _list_field(item, "unconfirmed_inferred_claims", context)
            )
        ]
        if any(
            row["source"] != IntentSource.INFERRED.value
            or row["claim_state"] != IntentClaimState.ACTIVE.value
            for row in unconfirmed_claims
        ):
            raise ValueError(
                f"{context}.unconfirmed_inferred_claims must contain only active inferred claims"
            )
        result["unconfirmed_inferred_claims"] = unconfirmed_claims
    return result


def _review_brief_intent_claim(value: Any, context: str) -> dict[str, Any]:
    claim = _intent_claim(value, context)
    return {
        "claim_id": claim.claim_id,
        "field": claim.field.value,
        "value": claim.value,
        "source": claim.source.value,
        "origin": claim.origin.value,
        "confidence": claim.confidence.value,
        "source_refs": list(claim.source_refs),
        "evidence_refs": list(claim.evidence_refs),
        "claim_state": claim.claim_state.value,
        "conclusion_impact": claim.conclusion_impact.value,
    }


def _review_brief_clarification(value: Any, context: str) -> dict[str, Any]:
    question = _clarification_question(value, context)
    return {
        "question_id": question.question_id,
        "field": question.field.value,
        "question": question.question,
        "rationale": question.rationale,
        "proposed_values": list(question.proposed_values),
        "claim_ids": list(question.claim_ids),
        "status": question.status.value,
        "user_response": question.user_response,
        "continuation_basis": question.continuation_basis,
        "resolved_values": list(question.resolved_values),
        "decision_id": question.decision_id,
    }


def _review_brief_risk_assessment(value: Any) -> dict[str, Any]:
    context = "review_brief.initial_and_final_risk_assessment"
    item = _object(value, context)
    _exact(item, {"initial", "final"}, context)
    return {
        "initial": _review_brief_initial_risk(item["initial"]),
        "final": _review_brief_final_risk(item["final"]),
    }


def _review_brief_initial_risk(value: Any) -> dict[str, Any]:
    context = "review_brief.initial_and_final_risk_assessment.initial"
    item = _object(value, context)
    _exact(
        item,
        {
            "level",
            "dimensions",
            "reasons",
            "signal_refs",
            "uncertainties",
            "suggested_focus",
        },
        context,
    )
    return {
        "level": _enum_field(RiskLevel, item, "level", context).value,
        "dimensions": _string_mapping(item, "dimensions", context),
        "reasons": _string_list(item, "reasons", context),
        "signal_refs": _string_list(item, "signal_refs", context),
        "uncertainties": _string_list(item, "uncertainties", context),
        "suggested_focus": _string_list(item, "suggested_focus", context),
    }


def _review_brief_final_risk(value: Any) -> dict[str, Any]:
    context = "review_brief.initial_and_final_risk_assessment.final"
    item = _object(value, context)
    abbreviated_fields = {"status", "level", "reasons"}
    if set(item) == abbreviated_fields:
        return {
            "status": _string(item, "status", context),
            "level": _enum_field(RiskLevel, item, "level", context).value,
            "reasons": _string_list(item, "reasons", context),
        }
    canonical = final_risk_to_dict(final_risk_from_dict(item))
    if not {
        "applied_memory",
        "memory_diagnostics",
        "residual_risk",
    }.intersection(item):
        for field_name in (
            "applied_memory",
            "memory_diagnostics",
            "residual_risk",
        ):
            canonical.pop(field_name, None)
    return canonical


def _review_brief_quality_gates(value: Any) -> list[dict[str, Any]]:
    context = "review_brief.quality_gates"
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        row_context = f"{context}[{index}]"
        item = _object(row, row_context)
        _required_with_optional(
            item,
            {"name", "status", "command", "summary", "observation_ref"},
            {
                "category",
                "cost",
                "source",
                "blocking",
                "reason",
                "exit_code",
                "duration_seconds",
                "output_truncated",
                "sandbox",
            },
            row_context,
        )
        result = _quality_gate_result_from_mapping(item, row_context)
        rows.append(
            {
                "name": result.name,
                "status": result.status,
                "command": list(result.command),
                "summary": result.summary,
                "observation_ref": result.observation_ref,
                "category": result.category,
                "cost": result.cost,
                "source": result.source,
                "blocking": result.blocking,
                "reason": result.reason,
                "exit_code": result.exit_code,
                "duration_seconds": result.duration_seconds,
                "output_truncated": result.output_truncated,
                "sandbox": result.sandbox,
            }
        )
    return rows


def _review_brief_change_map(value: Any) -> dict[str, Any]:
    context = "review_brief.change_map_and_repository_impact"
    item = _object(value, context)
    required = {
        "changed_files",
        "repository_intelligence_summary",
        "observation_count",
        "reviewer_summary",
    }
    missing = required - set(item)
    if missing:
        raise ValueError(
            f"{context} is missing required field(s): {', '.join(sorted(missing))}"
        )
    unexpected = set(item) - required - {"incremental_priority"}
    if unexpected:
        raise ValueError(
            f"{context} contains unsupported field(s): "
            f"{', '.join(sorted(str(name) for name in unexpected))}"
        )
    result = {
        "changed_files": _string_list(item, "changed_files", context),
        "repository_intelligence_summary": _string(
            item,
            "repository_intelligence_summary",
            context,
        ),
        "observation_count": _non_negative_integer(
            item,
            "observation_count",
            context,
        ),
        "reviewer_summary": _review_brief_reviewer_summary(
            item["reviewer_summary"]
        ),
    }
    if "incremental_priority" in item:
        result["incremental_priority"] = incremental_priority_to_dict(
            incremental_priority_from_dict(
                _object(
                    item["incremental_priority"],
                    f"{context}.incremental_priority",
                )
            )
        )
    return result


def _review_brief_reviewer_summary(value: Any) -> dict[str, Any]:
    context = "review_brief.change_map_and_repository_impact.reviewer_summary"
    item = _object(value, context)
    if not item:
        return {}
    _required_with_optional(
        item,
        {"reviewer_count", "status_counts"},
        {
            "roles",
            "single_reviewer_summary",
            "termination_counts",
            "executions",
        },
        context,
    )
    result: dict[str, Any] = {
        "reviewer_count": _non_negative_integer(item, "reviewer_count", context),
        "status_counts": _non_negative_integer_mapping(
            item,
            "status_counts",
            context,
        ),
    }
    if "roles" in item:
        result["roles"] = _string_list(item, "roles", context)
    if "single_reviewer_summary" in item:
        result["single_reviewer_summary"] = _string(
            item,
            "single_reviewer_summary",
            context,
        )
    if "termination_counts" in item:
        result["termination_counts"] = _non_negative_integer_mapping(
            item,
            "termination_counts",
            context,
        )
    if "executions" in item:
        executions = item["executions"]
        if not isinstance(executions, list):
            raise ValueError(f"{context}.executions must be a list")
        result["executions"] = [
            _review_brief_reviewer_execution(
                execution,
                f"{context}.executions[{index}]",
            )
            for index, execution in enumerate(executions)
        ]
    return result


def _review_brief_reviewer_execution(value: Any, context: str) -> dict[str, Any]:
    item = _object(value, context)
    _exact(item, {"reviewer_index", "role", "status", "runtime"}, context)
    return {
        "reviewer_index": _non_negative_integer(
            item,
            "reviewer_index",
            context,
        ),
        "role": _string(item, "role", context),
        "status": _enum_field(
            ReviewerResultStatus,
            item,
            "status",
            context,
        ).value,
        "runtime": reviewer_runtime_to_dict(
            _reviewer_runtime_from_dict(
                item["runtime"],
                f"{context}.runtime",
            )
        ),
    }


def _review_brief_contract_coverage(value: Any) -> list[dict[str, Any]]:
    context = "review_brief.review_contract_coverage"
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    rows: list[dict[str, Any]] = []
    full_fields = {
        "reviewer_index",
        "role",
        "contract",
        "status",
        "summary",
        "evidence_refs",
        "unsupported_evidence_refs",
    }
    abbreviated_fields = {"contract", "status", "summary", "evidence_refs"}
    for index, row in enumerate(value):
        row_context = f"{context}[{index}]"
        item = _object(row, row_context)
        if set(item) == abbreviated_fields:
            rows.append(
                {
                    "contract": _string(item, "contract", row_context),
                    "status": _string(item, "status", row_context),
                    "summary": _string(item, "summary", row_context),
                    "evidence_refs": _string_list(
                        item,
                        "evidence_refs",
                        row_context,
                    ),
                }
            )
            continue
        _exact(item, full_fields, row_context)
        rows.append(
            {
                "reviewer_index": _integer(item, "reviewer_index", row_context),
                "role": _string(item, "role", row_context),
                "contract": _string(item, "contract", row_context),
                "status": _string(item, "status", row_context),
                "summary": _string(item, "summary", row_context),
                "evidence_refs": _string_list(
                    item,
                    "evidence_refs",
                    row_context,
                ),
                "unsupported_evidence_refs": _string_list(
                    item,
                    "unsupported_evidence_refs",
                    row_context,
                ),
            }
        )
    return rows


def _review_brief_verification_evidence(value: Any) -> list[dict[str, Any]]:
    context = "review_brief.verification_evidence"
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        row_context = f"{context}[{index}]"
        item = _object(row, row_context)
        if "kind" not in item:
            raise ValueError(f"{row_context} is missing required field(s): kind")
        kind = _string(item, "kind", row_context)
        if kind == "quality_gate":
            _required_with_optional(
                item,
                {
                    "kind",
                    "name",
                    "status",
                    "summary",
                    "command",
                    "observation_ref",
                },
                {
                    "category",
                    "cost",
                    "source",
                    "blocking",
                    "reason",
                    "duration_seconds",
                },
                row_context,
            )
            gate = _quality_gate_result_from_mapping(
                {key: value for key, value in item.items() if key != "kind"},
                row_context,
            )
            rows.append(
                {
                    "kind": kind,
                    "name": gate.name,
                    "status": gate.status,
                    "summary": gate.summary,
                    "command": list(gate.command),
                    "observation_ref": gate.observation_ref,
                    "category": gate.category,
                    "cost": gate.cost,
                    "source": gate.source,
                    "blocking": gate.blocking,
                    "reason": gate.reason,
                    "duration_seconds": gate.duration_seconds,
                }
            )
            continue
        if kind == "observation":
            _exact(item, {"kind", "id", "summary"}, row_context)
            rows.append(
                {
                    "kind": kind,
                    "id": _string(item, "id", row_context),
                    "summary": _string(item, "summary", row_context),
                }
            )
            continue
        raise ValueError(f"{row_context}.kind has unsupported value: {kind}")
    return rows


def _assignment_from_dict(value: Any, context: str) -> Assignment:
    item = _object(value, context)
    _required_with_optional(
        item,
        {
            "role",
            "mission",
            "assignment_reason",
            "assigned_contract",
            "required_checks",
            "initial_context",
            "max_turns",
            "max_tool_calls",
            "repository_permission",
            "command_permission",
        },
        {
            "max_output_tokens",
            "max_total_tokens",
            "max_elapsed_seconds",
            "max_provider_attempts",
            "assignment_id",
            "role_kind",
            "perspective_key",
            "planner_source",
        },
        context,
    )
    initial_payload = _object_field(item, "initial_context", context)
    initial_context = f"{context}.initial_context"
    _required_with_optional(
        initial_payload,
        {
            "changed_files",
            "diff_ranges",
            "code_ranges",
            "quality_gate_summary",
            "observation_refs",
        },
        {
            "signal_refs",
            "selected_memory_refs",
            "verification_template_hints",
        },
        initial_context,
    )
    return Assignment(
        role=_string(item, "role", context),
        mission=_string(item, "mission", context),
        assignment_reason=_string_list(item, "assignment_reason", context),
        assigned_contract=_string_list(item, "assigned_contract", context),
        required_checks=_string_list(item, "required_checks", context),
        initial_context=InitialContext(
            changed_files=_string_list(initial_payload, "changed_files", initial_context),
            diff_ranges=_string_list(initial_payload, "diff_ranges", initial_context),
            code_ranges=_string_list(initial_payload, "code_ranges", initial_context),
            quality_gate_summary=_string_mapping(
                initial_payload,
                "quality_gate_summary",
                initial_context,
            ),
            observation_refs=_string_list(
                initial_payload,
                "observation_refs",
                initial_context,
            ),
            signal_refs=(
                _string_list(initial_payload, "signal_refs", initial_context)
                if "signal_refs" in initial_payload
                else []
            ),
            selected_memory_refs=(
                _string_list(
                    initial_payload,
                    "selected_memory_refs",
                    initial_context,
                )
                if "selected_memory_refs" in initial_payload
                else []
            ),
            verification_template_hints=(
                _string_list(
                    initial_payload,
                    "verification_template_hints",
                    initial_context,
                )
                if "verification_template_hints" in initial_payload
                else []
            ),
        ),
        max_turns=_integer(item, "max_turns", context),
        max_tool_calls=_integer(item, "max_tool_calls", context),
        max_output_tokens=(
            _positive_integer(item, "max_output_tokens", context)
            if "max_output_tokens" in item
            else DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS
        ),
        max_total_tokens=(
            _positive_integer(item, "max_total_tokens", context)
            if "max_total_tokens" in item
            else DEFAULT_REVIEWER_MAX_TOTAL_TOKENS
        ),
        max_elapsed_seconds=(
            _positive_number(item, "max_elapsed_seconds", context)
            if "max_elapsed_seconds" in item
            else DEFAULT_REVIEWER_MAX_ELAPSED_SECONDS
        ),
        max_provider_attempts=(
            _positive_integer(item, "max_provider_attempts", context)
            if "max_provider_attempts" in item
            else DEFAULT_REVIEWER_MAX_PROVIDER_ATTEMPTS
        ),
        repository_permission=_string(item, "repository_permission", context),
        command_permission=_string(item, "command_permission", context),
        assignment_id=(
            _string(item, "assignment_id", context)
            if "assignment_id" in item
            else ""
        ),
        role_kind=(
            _string(item, "role_kind", context)
            if "role_kind" in item
            else "legacy"
        ),
        perspective_key=(
            _string(item, "perspective_key", context)
            if "perspective_key" in item
            else "legacy"
        ),
        planner_source=(
            _string(item, "planner_source", context)
            if "planner_source" in item
            else "legacy"
        ),
    )


def _model_envelope_from_dict(payload: Mapping[str, Any]) -> ModelInvocationEnvelope:
    item = _object(payload, "reviewer_envelope")
    _exact(item, {"system", "tools", "messages", "parameters"}, "reviewer_envelope")
    return ModelInvocationEnvelope(
        system=_string(item, "system", "reviewer_envelope"),
        tools=_object_list(item, "tools", "reviewer_envelope"),
        messages=_object_list(item, "messages", "reviewer_envelope"),
        parameters=dict(_object_field(item, "parameters", "reviewer_envelope")),
    )


def _model_response_from_dict(payload: Mapping[str, Any]) -> ModelResponse:
    item = _object(payload, "reviewer_response")
    _required_with_optional(
        item,
        {"content", "provider_name", "model", "raw"},
        {"runtime"},
        "reviewer_response",
    )
    return ModelResponse(
        content=_string(item, "content", "reviewer_response"),
        provider_name=_string(item, "provider_name", "reviewer_response"),
        model=_string(item, "model", "reviewer_response"),
        raw=dict(_object_field(item, "raw", "reviewer_response")),
    )


def _reviewer_runtime_from_response(
    payload: Mapping[str, Any],
) -> ReviewerRuntimeMetadata:
    item = _object(payload, "reviewer_response")
    if "runtime" not in item:
        return ReviewerRuntimeMetadata()

    return _reviewer_runtime_from_dict(
        item["runtime"],
        "reviewer_response.runtime",
    )


def _reviewer_runtime_from_dict(
    value: Any,
    context: str,
) -> ReviewerRuntimeMetadata:
    runtime = _object(value, context)
    _exact(
        runtime,
        {
            "provider_attempts",
            "model_turns",
            "tool_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "usage_available",
            "elapsed_seconds",
            "termination_reason",
        },
        context,
    )
    return ReviewerRuntimeMetadata(
        provider_attempts=_non_negative_integer(runtime, "provider_attempts", context),
        model_turns=_non_negative_integer(runtime, "model_turns", context),
        tool_calls=_non_negative_integer(runtime, "tool_calls", context),
        input_tokens=_non_negative_integer(runtime, "input_tokens", context),
        output_tokens=_non_negative_integer(runtime, "output_tokens", context),
        total_tokens=_non_negative_integer(runtime, "total_tokens", context),
        usage_available=_boolean(runtime, "usage_available", context),
        elapsed_seconds=_non_negative_number(runtime, "elapsed_seconds", context),
        termination_reason=_enum_field(
            ReviewerTerminationReason,
            runtime,
            "termination_reason",
            context,
        ),
    )


def _canonical_finding(value: Any, context: str) -> CanonicalFinding:
    item = _object(value, context)
    _required_with_optional(
        item,
        {"claim", "severity", "confidence", "evidence_refs", "reviewer_indices", "roles", "suggested_action"},
        {"path", "line", "impact", "verification_performed", "finding_id"},
        context,
    )
    line = item.get("line")
    if line is not None and (type(line) is not int or line < 1):
        raise ValueError(f"{context}.line must be a positive integer or null")
    return CanonicalFinding(
        claim=_string(item, "claim", context),
        severity=_string(item, "severity", context),
        confidence=_string(item, "confidence", context),
        evidence_refs=_string_list(item, "evidence_refs", context),
        reviewer_indices=_integer_list(item, "reviewer_indices", context),
        roles=_string_list(item, "roles", context),
        suggested_action=_optional_string(item, "suggested_action", context),
        path=(
            _optional_string(item, "path", context)
            if "path" in item
            else None
        ),
        line=line,
        impact=_string(item, "impact", context) if "impact" in item else "",
        verification_performed=(
            _string_list(item, "verification_performed", context)
            if "verification_performed" in item
            else []
        ),
        finding_id=(
            _string(item, "finding_id", context)
            if "finding_id" in item
            else None
        ),
    )


def _rejected_finding(value: Any, context: str) -> RejectedFinding:
    item = _object(value, context)
    _exact(
        item,
        {"reviewer_index", "role", "claim", "reason", "evidence_refs", "missing_evidence_refs"},
        context,
    )
    return RejectedFinding(
        reviewer_index=_integer(item, "reviewer_index", context),
        role=_string(item, "role", context),
        claim=_string(item, "claim", context),
        reason=_string(item, "reason", context),
        evidence_refs=_string_list(item, "evidence_refs", context),
        missing_evidence_refs=_string_list(item, "missing_evidence_refs", context),
    )


def _contract_coverage(value: Any, context: str) -> ContractCoverage:
    item = _object(value, context)
    _exact(
        item,
        {"reviewer_index", "role", "contract", "status", "summary", "evidence_refs", "unsupported_evidence_refs"},
        context,
    )
    return ContractCoverage(
        reviewer_index=_integer(item, "reviewer_index", context),
        role=_string(item, "role", context),
        contract=_string(item, "contract", context),
        status=_string(item, "status", context),
        summary=_string(item, "summary", context),
        evidence_refs=_string_list(item, "evidence_refs", context),
        unsupported_evidence_refs=_string_list(
            item,
            "unsupported_evidence_refs",
            context,
        ),
    )


def _brief_finding(value: Any, context: str) -> BriefFinding:
    item = _object(value, context)
    _required_with_optional(
        item,
        {"claim", "severity", "confidence", "evidence_refs", "reviewer_indices", "roles", "suggested_action"},
        {"path", "line", "impact", "verification_performed", "finding_id"},
        context,
    )
    line = item.get("line")
    if line is not None and (type(line) is not int or line < 1):
        raise ValueError(f"{context}.line must be a positive integer or null")
    return BriefFinding(
        claim=_string(item, "claim", context),
        severity=_string(item, "severity", context),
        confidence=_string(item, "confidence", context),
        evidence_refs=_string_list(item, "evidence_refs", context),
        reviewer_indices=_integer_list(item, "reviewer_indices", context),
        roles=_string_list(item, "roles", context),
        suggested_action=_optional_string(item, "suggested_action", context),
        path=(
            _optional_string(item, "path", context)
            if "path" in item
            else None
        ),
        line=line,
        impact=_string(item, "impact", context) if "impact" in item else "",
        verification_performed=(
            _string_list(item, "verification_performed", context)
            if "verification_performed" in item
            else []
        ),
        finding_id=(
            _string(item, "finding_id", context)
            if "finding_id" in item
            else None
        ),
    )


def _rejected_hypothesis(value: Any, context: str) -> RejectedHypothesis:
    item = _object(value, context)
    _exact(
        item,
        {"claim", "reason", "evidence_refs", "reviewer_index", "role"},
        context,
    )
    reviewer_index = item["reviewer_index"]
    if reviewer_index is not None and type(reviewer_index) is not int:
        raise ValueError(f"{context}.reviewer_index must be an integer or null")
    return RejectedHypothesis(
        claim=_string(item, "claim", context),
        reason=_string(item, "reason", context),
        evidence_refs=_string_list(item, "evidence_refs", context),
        reviewer_index=reviewer_index,
        role=_optional_string(item, "role", context),
    )


EnumType = TypeVar("EnumType")


def _enum_field(enum_type: type[EnumType], payload: Mapping[str, Any], field: str, context: str) -> EnumType:
    return _enum(enum_type, payload[field], f"{context}.{field}")


def _enum(enum_type: type[EnumType], value: Any, context: str) -> EnumType:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except ValueError as error:
        raise ValueError(f"{context} has unsupported value: {value}") from error


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _object_field(payload: Mapping[str, Any], field: str, context: str) -> Mapping[str, Any]:
    return _object(payload[field], f"{context}.{field}")


def _exact(payload: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = expected - set(payload)
    if missing:
        raise ValueError(f"{context} is missing required field(s): {', '.join(sorted(missing))}")
    unexpected = set(payload) - expected
    if unexpected:
        raise ValueError(f"{context} contains unsupported field(s): {', '.join(sorted(str(key) for key in unexpected))}")


def _required_with_optional(
    payload: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"{context} is missing required field(s): {', '.join(sorted(missing))}"
        )
    unexpected = set(payload) - required - optional
    if unexpected:
        raise ValueError(
            f"{context} contains unsupported field(s): "
            f"{', '.join(sorted(str(key) for key in unexpected))}"
        )


def _string(payload: Mapping[str, Any], field: str, context: str) -> str:
    value = payload[field]
    if not isinstance(value, str):
        raise ValueError(f"{context}.{field} must be a string")
    return value


def _optional_string(payload: Mapping[str, Any], field: str, context: str) -> str | None:
    value = payload[field]
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{context}.{field} must be a string or null")
    return value


def _integer(payload: Mapping[str, Any], field: str, context: str) -> int:
    value = payload[field]
    if type(value) is not int:
        raise ValueError(f"{context}.{field} must be an integer")
    return value


def _positive_integer(payload: Mapping[str, Any], field: str, context: str) -> int:
    value = _integer(payload, field, context)
    if value <= 0:
        raise ValueError(f"{context}.{field} must be a positive integer")
    return value


def _non_negative_integer(
    payload: Mapping[str, Any],
    field: str,
    context: str,
) -> int:
    value = _integer(payload, field, context)
    if value < 0:
        raise ValueError(f"{context}.{field} must be a non-negative integer")
    return value


def _positive_number(payload: Mapping[str, Any], field: str, context: str) -> float:
    value = payload[field]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{context}.{field} must be a positive number")
    return float(value)


def _non_negative_number(
    payload: Mapping[str, Any],
    field: str,
    context: str,
) -> float:
    value = payload[field]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{context}.{field} must be a non-negative number")
    return float(value)


def _boolean(payload: Mapping[str, Any], field: str, context: str) -> bool:
    value = payload[field]
    if type(value) is not bool:
        raise ValueError(f"{context}.{field} must be a boolean")
    return value


def _list_field(payload: Mapping[str, Any], field: str, context: str) -> list[Any]:
    value = payload[field]
    if not isinstance(value, list):
        raise ValueError(f"{context}.{field} must be a list")
    return value


def _string_list(payload: Mapping[str, Any], field: str, context: str) -> list[str]:
    value = _list_field(payload, field, context)
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{context}.{field} must contain only strings")
    return list(value)


def _integer_list(payload: Mapping[str, Any], field: str, context: str) -> list[int]:
    value = _list_field(payload, field, context)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{context}.{field} must contain only integers")
    return list(value)


def _string_mapping(payload: Mapping[str, Any], field: str, context: str) -> dict[str, str]:
    value = _object_field(payload, field, context)
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        raise ValueError(f"{context}.{field} must map strings to strings")
    return dict(value)


def _non_negative_integer_mapping(
    payload: Mapping[str, Any],
    field: str,
    context: str,
) -> dict[str, int]:
    value = _object_field(payload, field, context)
    if any(
        not isinstance(key, str) or type(item) is not int or item < 0
        for key, item in value.items()
    ):
        raise ValueError(
            f"{context}.{field} must map strings to non-negative integers"
        )
    return dict(value)


def _object_list(payload: Mapping[str, Any], field: str, context: str) -> list[dict[str, Any]]:
    rows = _list_field(payload, field, context)
    return [dict(_object(row, f"{context}.{field}[{index}]")) for index, row in enumerate(rows)]
