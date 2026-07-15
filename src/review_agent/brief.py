from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import json
import math
import re
import sys
from types import MappingProxyType
from typing import Any

from review_agent.memory_curator import (
    CuratorDecisionOutcome,
    CuratorWarningCode,
    MemoryCandidateBatch,
    MemoryCuratorDecision,
    MemoryCuratorResult,
)
from review_agent.memory_models import (
    Applicability,
    CandidateStatus,
    DurableMemoryRecord,
    FeedbackCalibrationSummary,
    MemoryCandidate,
    MemoryKind,
    MemorySelectionDecision,
    MemoryScope,
    MemorySnapshot,
    PolicyEffect,
    PolicyEffectKind,
    RecordStatus,
    GitCommitSourceRef,
    HumanDeclarationSourceRef,
    ObservationSourceRef,
    RepositoryKnowledgeKey,
    RepositoryRangeSourceRef,
    RepositorySymbolSourceRef,
    RepositoryKnowledgeEntry,
    SessionArtifactSourceRef,
    SourceRefType,
    SUPPORTED_MEMORY_SELECTION_POLICY_VERSIONS,
    ValidityPolicy,
)
from review_agent.memory_policy import (
    PolicyCompilation,
    PolicyDiagnosticCode,
    PolicyDiagnosticSeverity,
    PolicyDisposition,
    RuntimeActionKind,
)
from review_agent.models import (
    ClarificationQuestion,
    ClarificationStatus,
    IntentClaim,
    IntentClaimState,
    IntentPacket,
    IntentSource,
    QualityGateResult,
    ReviewerResult,
    RiskAssessment,
)
from review_agent.repository_cache import (
    RepositoryCacheProvenance,
    RepositoryCacheResult,
    RepositoryCacheStatus,
)


@dataclass(frozen=True)
class BriefFinding:
    claim: str
    severity: str
    confidence: str
    evidence_refs: list[str]
    reviewer_indices: list[int] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    suggested_action: str | None = None
    path: str | None = None
    line: int | None = None
    impact: str = ""
    verification_performed: list[str] = field(default_factory=list)
    finding_id: str | None = None


@dataclass(frozen=True)
class RejectedHypothesis:
    claim: str
    reason: str
    evidence_refs: list[str] = field(default_factory=list)
    reviewer_index: int | None = None
    role: str | None = None


@dataclass(frozen=True)
class ReviewBrief:
    review_id: str
    base_revision: str
    head_revision: str
    change_intent: dict[str, Any]
    intent_assessment: dict[str, Any]
    initial_and_final_risk_assessment: dict[str, Any]
    quality_gates: list[dict[str, Any]]
    change_map_and_repository_impact: dict[str, Any]
    verified_findings: list[BriefFinding]
    rejected_hypotheses: list[RejectedHypothesis]
    uncertainties: list[str]
    reviewer_disagreements: list[str]
    review_contract_coverage: list[dict[str, Any]]
    verification_evidence: list[dict[str, Any]]
    human_review_checklist_and_reading_order: list[str]
    non_binding_recommendation: str
    orchestration: dict[str, Any] = field(default_factory=dict)
    semantic_reconciliation: dict[str, Any] = field(default_factory=dict)
    memory_audit: dict[str, Any] = field(default_factory=dict)

    @property
    def memory(self) -> dict[str, Any]:
        """Compatibility spelling for the optional bounded Memory audit."""

        return self.memory_audit


def build_review_brief(
    *,
    review_id: str,
    base_revision: str,
    head_revision: str,
    intent_packet: IntentPacket,
    risk_assessment: RiskAssessment,
    changed_files: list[str],
    quality_results: list[QualityGateResult],
    observation_summaries: dict[str, str] | None = None,
    repository_intelligence_summary: str | None = None,
    reviewer_result: ReviewerResult | None = None,
    multi_reviewer_summary: dict[str, object] | None = None,
    reconciliation_payload: dict[str, Any] | None = None,
    completion_summary: dict[str, Any] | None = None,
    final_risk_assessment: dict[str, Any] | None = None,
    incremental_priority: dict[str, Any] | None = None,
    planning_summary: dict[str, Any] | None = None,
    semantic_reconciliation_payload: dict[str, Any] | None = None,
    memory_audit_payload: Any | None = None,
    memory_snapshot: Any | None = None,
    compiled_memory_policy: Any | None = None,
    policy_compilation: Any | None = None,
    cache_provenance: Any | None = None,
    feedback_summary: Any | None = None,
    pending_memory_candidates: Any | None = None,
    memory_status: Any | None = None,
    memory_warnings: Any | None = None,
    curator_status: Any | None = None,
    outbox_status: Any | None = None,
) -> ReviewBrief:
    observations = observation_summaries or {}
    reconciliation = reconciliation_payload or {}
    completion = completion_summary or {}
    verified_findings = _verified_findings(reconciliation)
    rejected_hypotheses = _rejected_hypotheses(reconciliation, reviewer_result)
    uncertainties = _uncertainties(intent_packet, risk_assessment, reviewer_result, completion)
    if planning_summary is not None:
        planning_uncertainties = planning_summary.get("uncertainties", [])
        if isinstance(planning_uncertainties, list):
            uncertainties = _dedupe(
                [
                    *uncertainties,
                    *(
                        str(item)
                        for item in planning_uncertainties
                        if str(item).strip()
                    ),
                ]
            )

    change_map: dict[str, Any] = {
        "changed_files": list(changed_files),
        "repository_intelligence_summary": repository_intelligence_summary or "",
        "observation_count": len(observations),
        "reviewer_summary": _reviewer_summary(multi_reviewer_summary, reviewer_result),
    }
    if incremental_priority is not None:
        change_map["incremental_priority"] = dict(incremental_priority)

    return ReviewBrief(
        review_id=review_id,
        base_revision=base_revision,
        head_revision=head_revision,
        change_intent={
            "goal": intent_packet.goal,
            "acceptance_criteria": list(intent_packet.acceptance_criteria),
            "scope": list(intent_packet.scope),
            "constraints": list(intent_packet.constraints),
            "sources": {key: value.value for key, value in intent_packet.sources.items()},
            "provenance": [
                _intent_claim_to_dict(claim) for claim in intent_packet.provenance
            ],
        },
        intent_assessment={
            "status": intent_packet.status.value,
            "uncertainties": list(intent_packet.uncertainties),
            "source_counts": _source_counts(intent_packet),
            "clarification_history": [
                _clarification_to_dict(question)
                for question in intent_packet.clarifications
            ],
            "unresolved_questions": [
                _clarification_to_dict(question)
                for question in intent_packet.clarifications
                if question.status
                in {ClarificationStatus.PENDING, ClarificationStatus.OPEN}
            ],
            "unconfirmed_inferred_claims": [
                _intent_claim_to_dict(claim)
                for claim in intent_packet.provenance
                if claim.source is IntentSource.INFERRED
                and claim.claim_state is IntentClaimState.ACTIVE
            ],
        },
        initial_and_final_risk_assessment={
            "initial": _risk_to_dict(risk_assessment),
            "final": final_risk_assessment
            or {
                "status": "not_reassessed",
                "level": risk_assessment.level.value,
                "reasons": ["Final risk reassessment has not run in the local M1 path."],
            },
        },
        quality_gates=[_quality_result_to_dict(result) for result in quality_results],
        change_map_and_repository_impact=change_map,
        verified_findings=verified_findings,
        rejected_hypotheses=rejected_hypotheses,
        uncertainties=uncertainties,
        reviewer_disagreements=[str(item) for item in reconciliation.get("remaining_disagreements", [])],
        review_contract_coverage=_contract_coverage(reconciliation, reviewer_result),
        verification_evidence=_verification_evidence(quality_results, observations),
        human_review_checklist_and_reading_order=_human_review_checklist(
            changed_files=changed_files,
            risk_assessment=risk_assessment,
            verified_findings=verified_findings,
            uncertainties=uncertainties,
        ),
        non_binding_recommendation=str(completion.get("recommendation", "manual_review")),
        orchestration=dict(planning_summary or {}),
        semantic_reconciliation=dict(semantic_reconciliation_payload or {}),
        memory_audit=build_memory_audit_projection(
            memory_audit_payload,
            snapshot=memory_snapshot,
            compiled_policy=(
                compiled_memory_policy
                if compiled_memory_policy is not None
                else policy_compilation
            ),
            cache_provenance=cache_provenance,
            feedback_summary=feedback_summary,
            pending_candidates=pending_memory_candidates,
            status=memory_status,
            warnings=memory_warnings,
            curator=curator_status,
            outbox=outbox_status,
        ),
    )


def review_brief_to_dict(brief: ReviewBrief) -> dict[str, Any]:
    # ``asdict`` deep-copies every field before a caller can remove one.  Keep
    # the Memory sidecar out of that walk entirely: a canonical Snapshot may be
    # large and must cross only the bounded projection below.
    raw = {
        name: value
        for name, value in vars(brief).items()
        if name != "memory_audit"
    }
    payload = _json_ready(raw)
    for finding in payload["verified_findings"]:
        if finding.get("finding_id") is None:
            finding.pop("finding_id", None)
    if not brief.semantic_reconciliation:
        payload.pop("semantic_reconciliation", None)
    memory_audit = build_memory_audit_projection(brief.memory_audit)
    if memory_audit:
        payload["memory_audit"] = memory_audit
    else:
        payload.pop("memory_audit", None)
    return payload


def _verified_findings(reconciliation: dict[str, Any]) -> list[BriefFinding]:
    findings: list[BriefFinding] = []
    for item in reconciliation.get("canonical_findings", []):
        row = dict(item)
        findings.append(
            BriefFinding(
                claim=str(row.get("claim", "")),
                severity=str(row.get("severity", "")),
                confidence=str(row.get("confidence", "")),
                evidence_refs=[str(ref) for ref in row.get("evidence_refs", [])],
                reviewer_indices=[int(index) for index in row.get("reviewer_indices", [])],
                roles=[str(role) for role in row.get("roles", [])],
                suggested_action=str(row["suggested_action"]) if row.get("suggested_action") is not None else None,
                path=str(row["path"]) if row.get("path") is not None else None,
                line=int(row["line"]) if row.get("line") is not None else None,
                impact=str(row.get("impact", "")),
                verification_performed=[
                    str(item) for item in row.get("verification_performed", [])
                ],
                finding_id=(
                    str(row["finding_id"])
                    if row.get("finding_id") is not None
                    else None
                ),
            )
        )
    return findings


def _rejected_hypotheses(
    reconciliation: dict[str, Any],
    reviewer_result: ReviewerResult | None,
) -> list[RejectedHypothesis]:
    rejected: list[RejectedHypothesis] = []
    for item in reconciliation.get("rejected_findings", []):
        row = dict(item)
        rejected.append(
            RejectedHypothesis(
                claim=str(row.get("claim", "")),
                reason=str(row.get("reason", "unsupported_claim")),
                evidence_refs=[str(ref) for ref in row.get("evidence_refs", [])],
                reviewer_index=int(row["reviewer_index"]) if row.get("reviewer_index") is not None else None,
                role=str(row["role"]) if row.get("role") is not None else None,
            )
        )
    if reviewer_result is not None:
        rejected.extend(
            RejectedHypothesis(claim=str(item), reason="reviewer_rejected_hypothesis")
            for item in reviewer_result.rejected_hypotheses
        )
    return rejected


def _uncertainties(
    intent_packet: IntentPacket,
    risk_assessment: RiskAssessment,
    reviewer_result: ReviewerResult | None,
    completion: dict[str, Any],
) -> list[str]:
    items: list[str] = []
    items.extend(intent_packet.uncertainties)
    items.extend(risk_assessment.uncertainties)
    if reviewer_result is not None:
        items.extend(reviewer_result.uncertainties)
    items.extend(str(item) for item in completion.get("uncertainties", []))
    items.extend(str(item) for item in completion.get("blockers", []))
    items.extend(f"Missing perspective: {item}" for item in completion.get("missing_perspectives", []))
    return _dedupe(items)


def _contract_coverage(
    reconciliation: dict[str, Any],
    reviewer_result: ReviewerResult | None,
) -> list[dict[str, Any]]:
    if reconciliation.get("contract_coverage"):
        return [dict(item) for item in reconciliation["contract_coverage"]]
    if reviewer_result is None:
        return []
    return [
        {
            "contract": assessment.contract,
            "status": assessment.status.value,
            "summary": assessment.summary,
            "evidence_refs": list(assessment.evidence_refs),
        }
        for assessment in reviewer_result.contract_assessments
    ]


def _verification_evidence(
    quality_results: list[QualityGateResult],
    observations: dict[str, str],
) -> list[dict[str, Any]]:
    evidence = [
        {
            "kind": "quality_gate",
            "name": result.name,
            "status": result.status,
            "summary": result.summary,
            "command": list(result.command),
            "observation_ref": result.observation_ref,
            "category": result.category,
            "cost": result.cost,
            "source": result.source,
            "blocking": result.blocking,
            "reason": result.reason,
            "duration_seconds": result.duration_seconds,
        }
        for result in quality_results
    ]
    evidence.extend(
        {
            "kind": "observation",
            "id": observation_id,
            "summary": summary,
        }
        for observation_id, summary in observations.items()
    )
    return evidence


def _reviewer_summary(
    multi_reviewer_summary: dict[str, object] | None,
    reviewer_result: ReviewerResult | None,
) -> dict[str, object]:
    if multi_reviewer_summary:
        return dict(multi_reviewer_summary)
    if reviewer_result is None:
        return {}
    return {
        "reviewer_count": 1,
        "status_counts": {reviewer_result.status.value: 1},
        "single_reviewer_summary": reviewer_result.investigation_summary,
    }


def _human_review_checklist(
    *,
    changed_files: list[str],
    risk_assessment: RiskAssessment,
    verified_findings: list[BriefFinding],
    uncertainties: list[str],
) -> list[str]:
    checklist: list[str] = []
    checklist.extend(f"Read changed file: {path}" for path in changed_files)
    checklist.extend(f"Check review focus: {focus}" for focus in risk_assessment.suggested_focus)
    checklist.extend(f"Verify finding: {finding.claim}" for finding in verified_findings)
    checklist.extend(f"Resolve uncertainty: {uncertainty}" for uncertainty in uncertainties)
    return checklist or ["No prioritized human review items were generated."]


def _source_counts(intent_packet: IntentPacket) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in intent_packet.sources.values():
        counts[source.value] = counts.get(source.value, 0) + 1
    return counts


def _intent_claim_to_dict(claim: IntentClaim) -> dict[str, Any]:
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


def _clarification_to_dict(question: ClarificationQuestion) -> dict[str, Any]:
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


def _risk_to_dict(risk_assessment: RiskAssessment) -> dict[str, Any]:
    return {
        "level": risk_assessment.level.value,
        "dimensions": dict(risk_assessment.dimensions),
        "reasons": list(risk_assessment.reasons),
        "signal_refs": list(risk_assessment.signal_refs),
        "uncertainties": list(risk_assessment.uncertainties),
        "suggested_focus": list(risk_assessment.suggested_focus),
    }


def _quality_result_to_dict(result: QualityGateResult) -> dict[str, Any]:
    return {
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


_MEMORY_AUDIT_SCHEMA = "memory_audit_v1"

# These limits apply to the reporting boundary, not to the Memory Store.  A
# boundary overrun is represented by a fixed code and an empty safe payload;
# truncating an attacker-controlled string would still publish part of it.
_AUDIT_MAX_DEPTH = 8
_AUDIT_MAX_KEYS = 512
_AUDIT_MAX_ITEMS = 256
_AUDIT_MAX_TEXT = 8192
_AUDIT_MAX_JSON_BYTES = 128 * 1024
_AUDIT_MAX_INTEGER = (1 << 63) - 1
_AUDIT_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+#/@-]{0,255}$")
_AUDIT_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_AUDIT_GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_AUDIT_STABLE_ID = re.compile(r"^[A-Z][A-Z0-9]*-[0-9a-f]{64}$")
_AUDIT_FINDING_ID = re.compile(r"^F-[0-9a-f]{32}(?:[0-9a-f]{32})?$")
_AUDIT_OBSERVATION_ID = re.compile(r"^O-[0-9a-f]{12,64}$")
_AUDIT_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_AUDIT_MAPPING_PROXY_TYPE = type(MappingProxyType({}))

_AUDIT_LIMIT_CODES = frozenset(
    {
        "memory_audit_depth_exceeded",
        "memory_audit_keys_exceeded",
        "memory_audit_items_exceeded",
        "memory_audit_string_exceeded",
        "memory_audit_bytes_exceeded",
        "memory_audit_unsupported_input",
        "memory_audit_invalid_shape",
    }
)
_AUDIT_DEGRADATION_CODES = frozenset(
    {
        "memory_unavailable",
        "hard_policy_blocked",
        "outbox_pending",
        "curator_fallback",
        "curator_disabled",
        "cache_corruption",
        "cache_rebuild",
        "cache_miss",
        "audit_projection_degraded",
        "record_not_applied",
        "record_status_missing",
        "selection_missing",
        "selection_not_selected",
        "approval_missing",
        "compiled_provenance_missing",
        "compiled_provenance_not_authoritative",
        "invalid_record",
        "invalid_field",
        "store_corrupt",
        "store_unavailable",
        "unsupported_schema",
        "hash_mismatch",
        "missing_blob",
        "missing_entry",
        "unknown",
    }
)
_AUDIT_SAFE_REASON_CODES = frozenset(
    {
        "selected",
        "applicable",
        "target_revision_valid",
        "stage_kind_allowed",
        "stage_kind_not_allowed",
        "target_scope_does_not_match",
        "record_revalidation_required",
        "record_revoked",
        "record_superseded",
        "record_expired",
        "expiry_time_reached",
        "expiry_commit_reached",
        "expiry_condition_unresolved",
        "record_status_not_authoritative",
        "target_validity_unavailable",
        "target_head_missing",
        "target_precedes_valid_from",
        "diverged_lineage",
        "valid_from_missing",
        "source_missing",
        "source_changed",
        "scope_changed",
        "scope_change_unavailable",
        "record_budget",
        "per_kind_budget",
        "snapshot_byte_budget",
        "source_content_hash",
        "symbol_signature",
        "scope_change_trigger",
        "manual_until_revoked",
    }
)
_AUDIT_SAFE_CACHE_CORRUPTION = frozenset(
    {
        "corruption",
        "pinned_corrupt_entry",
        "missing_blob",
        "blob_missing",
        "hash_mismatch",
        "blob_hash_mismatch",
        "entry_missing",
        "manifest_mismatch",
        "unavailable",
        "unsupported_schema",
        "validation",
        "busy",
        "read_only",
        "migration",
        "not_found",
        "conflict",
        "rebuild_required",
    }
)
_AUDIT_SAFE_FALLBACK_VALUES = frozenset(
    {
        "available",
        "unavailable",
        "disabled",
        "failed",
        "timeout",
        "not_configured",
        "unknown",
        "none",
        "python_ast",
        "python_ast+git_grep",
        "lsp+python_ast+git_grep",
        "git_grep",
        "ripgrep",
    }
)
_AUDIT_SAFE_FALLBACK_KEYS = frozenset(
    {
        "lsp_status",
        "fallback_strategy",
        "strategy",
        "text_search_backend",
        "parser_status",
    }
)
_AUDIT_SAFE_OUTBOX_STATUSES = frozenset(
    {"pending", "outbox_pending", "persisted", "completed", "replayed", "failed", "skipped", "disabled"}
)
_AUDIT_SAFE_OUTBOX_CODES = frozenset(
    {
        "unavailable",
        "unsupported_schema",
        "corruption",
        "conflict",
        "busy",
        "validation",
        "not_found",
        "read_only",
        "migration",
        "request_replayed",
        "request_pending",
        "persistence_failed",
        "outbox_invalid",
    }
)
_AUDIT_SAFE_CANDIDATE_DEDUPE = frozenset(
    {
        "unique",
        "exact_replay",
        "active_duplicate",
        "pending_duplicate",
        "rejected_unchanged",
        "enhanced_provenance",
    }
)
_AUDIT_SAFE_CANDIDATE_PERSISTENCE = frozenset(
    {"persisted", "replayed"}
)
_AUDIT_SAFE_REVIEW_IMPACTS = frozenset(
    {"none", "no_change", "uncertainty_only", "manual_review"}
)


class _AuditBoundaryError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code if code in _AUDIT_LIMIT_CODES else "memory_audit_invalid_shape"


class _AuditBudget:
    def __init__(self) -> None:
        self.keys = 0
        self.items = 0
        self.json_bytes = 2

    def add_bytes(self, amount: int) -> None:
        self.json_bytes += amount
        if self.json_bytes > _AUDIT_MAX_JSON_BYTES:
            raise _AuditBoundaryError("memory_audit_bytes_exceeded")


def _feedback_wrapper_types() -> tuple[type[Any], ...]:
    """Resolve optional Feedback wrappers only after module import completes.

    ``memory_feedback`` imports hydration, and hydration imports this module.
    Looking the classes up lazily avoids making that dependency part of the
    import graph.  The returned values are exact types; no protocol or method
    discovery is used.
    """

    module = sys.modules.get("review_agent.memory_feedback")
    if module is None:
        return ()
    result: list[type[Any]] = []
    for name in ("FeedbackAggregationResult", "FeedbackStageProjection"):
        candidate = module.__dict__.get(name)
        if isinstance(candidate, type):
            result.append(candidate)
    return tuple(result)


def _canonical_audit_types() -> tuple[type[Any], ...]:
    return (
        MemorySnapshot,
        DurableMemoryRecord,
        MemorySelectionDecision,
        MemoryCandidate,
        MemoryScope,
        PolicyEffect,
        FeedbackCalibrationSummary,
        PolicyCompilation,
        RepositoryCacheProvenance,
        RepositoryCacheResult,
        RepositoryKnowledgeEntry,
        MemoryCuratorDecision,
        MemoryCuratorResult,
        MemoryCandidateBatch,
        RepositoryKnowledgeKey,
        GitCommitSourceRef,
        HumanDeclarationSourceRef,
        ObservationSourceRef,
        RepositoryRangeSourceRef,
        RepositorySymbolSourceRef,
        SessionArtifactSourceRef,
        *_feedback_wrapper_types(),
    )


def _known_enum_value(value: Any) -> Any:
    enum_types = (
        Applicability,
        CandidateStatus,
        CuratorDecisionOutcome,
        CuratorWarningCode,
        MemoryKind,
        PolicyDiagnosticCode,
        PolicyDiagnosticSeverity,
        PolicyDisposition,
        PolicyEffectKind,
        RecordStatus,
        RepositoryCacheStatus,
        RuntimeActionKind,
        SourceRefType,
        ValidityPolicy,
    )
    if type(value) in enum_types:
        return value.value
    return value


def _is_plain_mapping(value: Any) -> bool:
    return type(value) in {dict, _AUDIT_MAPPING_PROXY_TYPE}


def _bounded_audit_graph(
    value: Any,
    budget: _AuditBudget,
    *,
    depth: int = 0,
) -> Any:
    """Copy only a bounded plain graph, never invoking foreign object code."""

    if depth > _AUDIT_MAX_DEPTH:
        raise _AuditBoundaryError("memory_audit_depth_exceeded")
    value = _known_enum_value(value)
    if type(value) in _canonical_audit_types():
        return value
    if value is None:
        budget.add_bytes(4)
        return None
    if type(value) is bool:
        budget.add_bytes(5)
        return value
    if type(value) is int:
        if abs(value) > _AUDIT_MAX_INTEGER:
            raise _AuditBoundaryError("memory_audit_invalid_shape")
        budget.add_bytes(len(str(value)))
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _AuditBoundaryError("memory_audit_invalid_shape")
        budget.add_bytes(len(json.dumps(value, separators=(",", ":"))))
        return value
    if type(value) is str:
        if len(value) > _AUDIT_MAX_TEXT:
            raise _AuditBoundaryError("memory_audit_string_exceeded")
        budget.add_bytes(len(json.dumps(value, ensure_ascii=False).encode("utf-8")))
        return value
    if _is_plain_mapping(value):
        if len(value) > _AUDIT_MAX_KEYS:
            raise _AuditBoundaryError("memory_audit_keys_exceeded")
        budget.keys += len(value)
        if budget.keys > _AUDIT_MAX_KEYS:
            raise _AuditBoundaryError("memory_audit_keys_exceeded")
        result: dict[str, Any] = {}
        # ``dict`` and ``mappingproxy`` are the only accepted Mapping roots;
        # their iteration cannot dispatch to a caller-owned Store object.
        for key, item in value.items():
            if type(key) is not str:
                raise _AuditBoundaryError("memory_audit_invalid_shape")
            if len(key) > _AUDIT_MAX_TEXT:
                raise _AuditBoundaryError("memory_audit_string_exceeded")
            budget.add_bytes(len(json.dumps(key, ensure_ascii=False).encode("utf-8")))
            result[key] = _bounded_audit_graph(item, budget, depth=depth + 1)
        budget.add_bytes(2)
        return result
    if type(value) in {list, tuple}:
        if len(value) > _AUDIT_MAX_ITEMS:
            raise _AuditBoundaryError("memory_audit_items_exceeded")
        budget.items += len(value)
        if budget.items > _AUDIT_MAX_ITEMS:
            raise _AuditBoundaryError("memory_audit_items_exceeded")
        result = [
            _bounded_audit_graph(item, budget, depth=depth + 1)
            for item in value
        ]
        budget.add_bytes(2)
        return result
    # In particular, do not inspect ``to_dict``, dataclass fields, properties,
    # iterators, or ``__str__`` on an unknown object.
    raise _AuditBoundaryError("memory_audit_unsupported_input")


def _degraded_memory_audit(*codes: str) -> dict[str, Any]:
    reasons = sorted(
        {
            code if code in _AUDIT_LIMIT_CODES else "memory_audit_invalid_shape"
            for code in codes
        }
        | {"audit_projection_degraded"}
    )
    return {
        "schema_version": _MEMORY_AUDIT_SCHEMA,
        "applied_memory": [],
        "not_applied_memory": [],
        "compiled_policy": {},
        "cache_provenance": [],
        "warnings": [],
        "feedback_summary": {},
        "pending_candidates": [],
        "status": {
            "memory_unavailable": False,
            "hard_policy_blocked": False,
            "outbox_pending": False,
            "degraded": True,
            "degradation_reasons": reasons,
        },
    }


def _finalize_memory_audit(result: dict[str, Any]) -> dict[str, Any]:
    try:
        bounded = _bounded_audit_graph(result, _AuditBudget())
        encoded = json.dumps(
            bounded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > _AUDIT_MAX_JSON_BYTES:
            raise _AuditBoundaryError("memory_audit_bytes_exceeded")
        return bounded
    except _AuditBoundaryError as error:
        return _degraded_memory_audit(error.code)


def build_memory_audit_projection(
    payload: Any | None = None,
    *,
    snapshot: Any | None = None,
    compiled_policy: Any | None = None,
    cache_provenance: Any | None = None,
    feedback_summary: Any | None = None,
    pending_candidates: Any | None = None,
    status: Any | None = None,
    warnings: Any | None = None,
    curator: Any | None = None,
    outbox: Any | None = None,
) -> dict[str, Any]:
    """Return a bounded, content-free Memory audit projection.

    The accepted object boundary is deliberately closed: exact canonical
    Memory types are allowed, and wire input must be a plain ``dict`` (or a
    mapping proxy containing only bounded primitive/container values and exact
    canonical values).  No arbitrary object's conversion method is called.
    """

    if payload is None and all(
        value is None
        for value in (
            snapshot,
            compiled_policy,
            cache_provenance,
            feedback_summary,
            pending_candidates,
            status,
            warnings,
            curator,
            outbox,
        )
    ):
        return {}

    budget = _AuditBudget()
    try:
        prepared_payload = (
            None
            if payload is None
            else _bounded_audit_graph(payload, budget)
        )
        prepared = {
            name: (
                None
                if value is None
                else _bounded_audit_graph(value, budget)
            )
            for name, value in (
                ("snapshot", snapshot),
                ("compiled_policy", compiled_policy),
                ("cache_provenance", cache_provenance),
                ("feedback_summary", feedback_summary),
                ("pending_candidates", pending_candidates),
                ("status", status),
                ("warnings", warnings),
                ("curator", curator),
                ("outbox", outbox),
            )
        }
    except _AuditBoundaryError as error:
        return _degraded_memory_audit(error.code)

    if prepared_payload is None:
        root: dict[str, Any] = {}
    elif type(prepared_payload) is MemorySnapshot:
        root = {"snapshot": prepared_payload}
    elif type(prepared_payload) is MemoryCuratorResult:
        root = {
            "curator": prepared_payload,
            "pending_candidates": prepared_payload.batch,
        }
    elif type(prepared_payload) is MemoryCandidateBatch:
        root = {"pending_candidates": prepared_payload}
    elif type(prepared_payload) is MemoryCandidate:
        root = {"pending_candidates": [prepared_payload]}
    elif type(prepared_payload) is DurableMemoryRecord:
        root = {"snapshot": {"eligible_records": [prepared_payload]}}
    elif type(prepared_payload) is MemorySelectionDecision:
        root = {"snapshot": {"applicability_decisions": [prepared_payload]}}
    elif type(prepared_payload) is PolicyCompilation:
        root = {"compiled_policy": prepared_payload}
    elif type(prepared_payload) in {
        RepositoryCacheProvenance,
        RepositoryCacheResult,
        RepositoryKnowledgeEntry,
    }:
        root = {"cache_provenance": prepared_payload}
    elif type(prepared_payload) is FeedbackCalibrationSummary or type(
        prepared_payload
    ) in _feedback_wrapper_types():
        root = {"feedback_summary": prepared_payload}
    elif _is_plain_mapping(prepared_payload):
        root = dict(prepared_payload)
    else:
        return _degraded_memory_audit("memory_audit_unsupported_input")

    if root and "eligible_records" in root and not any(
        key in root
        for key in {
            "snapshot",
            "memory_snapshot",
            "applied_memory",
            "compiled_policy",
            "status",
        }
    ):
        root = {"snapshot": root}

    for name, value in prepared.items():
        if value is not None:
            root[name] = value

    snapshot_value = _audit_pick(root, "snapshot", "memory_snapshot")
    snapshot_map = _audit_mapping(snapshot_value)
    policy_value = _audit_pick(
        root,
        "compiled_policy",
        "policy_compilation",
        "memory_policy",
        "policy",
    )
    cache_value = _audit_pick(
        root,
        "cache_provenance",
        "repository_cache_provenance",
        "repository_knowledge_cache",
        "repository_knowledge",
        "cache",
    )
    feedback_value = _audit_pick(
        root,
        "feedback_summary",
        "feedback_calibration_summary",
        "memory_feedback_summary",
        "feedback",
    )
    if feedback_value is None:
        feedback_value = _audit_pick(
            snapshot_map,
            "feedback_summary",
            "feedback_calibration_summary",
            "memory_feedback_summary",
        )
    candidates_value = _audit_pick(
        root,
        "pending_candidates",
        "memory_candidates",
        "candidate_batch",
        "candidates",
    )
    status_value = _audit_pick(root, "status", "memory_status", "runtime_status")
    warnings_value = _audit_pick(
        root, "warnings", "validity_warnings", "memory_warnings"
    )
    curator_value = _audit_pick(
        root,
        "curator",
        "curator_status",
        "curator_decision",
        "curator_state",
    )
    outbox_value = _audit_pick(
        root,
        "outbox",
        "outbox_status",
        "memory_outbox",
        "outbox_state",
    )
    if status_value is None and any(
        key in root
        for key in {
            "available",
            "memory_unavailable",
            "hard_policy_blocked",
            "outbox_pending",
            "unavailable_reason",
            "degradation_reasons",
        }
    ):
        status_value = root
    status_map = _audit_mapping(status_value)
    if curator_value is None:
        curator_value = status_map.get("curator")
    if outbox_value is None:
        outbox_value = status_map.get("outbox")

    records_value = _audit_pick(
        snapshot_map,
        "eligible_records",
        "records",
        "applied_memory",
    )
    if records_value is None:
        records_value = root.get("applied_memory")
    decisions_value = _audit_pick(
        snapshot_map,
        "applicability_decisions",
        "decisions",
    )
    decisions = {
        row["memory_id"]: row
        for row in _audit_rows(decisions_value)
        if _audit_id(row.get("memory_id"), "MEM")
    }

    policy = _project_compiled_policy(policy_value)
    policy_provenance = {
        row["memory_id"]: row
        for row in policy.get("provenance", [])
        if row.get("memory_id")
    }
    applied: list[dict[str, Any]] = []
    not_applied: list[dict[str, Any]] = _project_not_applied_rows(
        root.get("not_applied_memory")
    )
    for item in _audit_values(records_value):
        row = _audit_mapping(item)
        if "record" in row:
            row = _audit_mapping(row.get("record"))
        decision = decisions.get(_audit_id(row.get("memory_id"), "MEM"))
        provenance = policy_provenance.get(_audit_id(row.get("memory_id"), "MEM"))
        failure = _authority_failure(row, decision, provenance)
        if failure is None:
            projected = _project_applied_memory(row, decision, provenance)
            if projected is not None:
                applied.append(projected)
        else:
            projected = _project_not_applied_memory(row, decision, failure)
            if projected is not None:
                not_applied.append(projected)
    applied = sorted(
        {row["memory_id"]: row for row in applied}.values(),
        key=lambda row: row["memory_id"],
    )[:_AUDIT_MAX_ITEMS]
    not_applied = _unique_audit_rows(not_applied)

    cache = _project_cache_provenance(cache_value)
    for reference in _audit_values(_audit_pick(snapshot_map, "repository_knowledge_refs")):
        entry_id = _audit_id(reference, "RKE")
        if entry_id and not any(row.get("entry_id") == entry_id for row in cache):
            cache.append({"entry_id": entry_id, "status": "referenced"})
    cache = sorted(
        {(
            row.get("entry_id", ""),
            row.get("status", ""),
            row.get("key_hash", ""),
        ): row for row in cache}.values(),
        key=lambda row: (row.get("entry_id", ""), row.get("status", "")),
    )[:_AUDIT_MAX_ITEMS]

    audit_warnings = _project_warnings(warnings_value)
    for row in decisions.values():
        applicability = _audit_enum(row.get("applicability"), Applicability)
        if not applicability or applicability == Applicability.SELECTED.value:
            continue
        category = _warning_category(applicability, row.get("reason_codes", []))
        if category is None:
            continue
        reason_codes = _safe_reason_list(row.get("reason_codes"))
        requires = _audit_bool(row.get("requires_revalidation"))
        audit_warnings.append(
            {
                "category": category,
                "memory_id": _audit_id(row.get("memory_id"), "MEM"),
                "applicability": applicability,
                "reason_codes": reason_codes,
                "requires_revalidation": (
                    True if requires is None and category == "revalidation" else requires
                ),
            }
        )
    audit_warnings = _unique_audit_rows(audit_warnings)

    feedback = _project_feedback_summary(feedback_value)
    candidates = _project_pending_candidates(candidates_value)

    curator_projection = _project_curator(curator_value)
    outbox_projection = _project_outbox(outbox_value)
    status_projection = _project_memory_status(
        status_value,
        policy=policy,
        curator=curator_projection,
        outbox=outbox_projection,
    )

    snapshot_projection = _project_snapshot(snapshot_map)
    meaningful = bool(
        applied
        or not_applied
        or policy
        or cache
        or audit_warnings
        or feedback
        or candidates
        or snapshot_projection
        or status_value is not None
        or curator_value is not None
        or outbox_value is not None
    )
    if not meaningful:
        return {}

    result: dict[str, Any] = {
        "schema_version": _MEMORY_AUDIT_SCHEMA,
        "applied_memory": applied,
        "not_applied_memory": not_applied,
        "compiled_policy": policy,
        "cache_provenance": cache,
        "warnings": audit_warnings,
        "feedback_summary": feedback,
        "pending_candidates": candidates,
        "status": status_projection,
    }
    if snapshot_projection:
        result["snapshot"] = snapshot_projection
    return _finalize_memory_audit(result)


def memory_audit_to_dict(payload: Any | None) -> dict[str, Any]:
    """Compatibility helper for callers that already hold an audit payload."""

    return build_memory_audit_projection(payload)


def _audit_mapping(value: Any) -> dict[str, Any]:
    """Map only exact canonical values or an already bounded plain mapping."""

    if _is_plain_mapping(value):
        return dict(value)
    converters: dict[type[Any], Any] = {
        MemorySnapshot: MemorySnapshot.to_dict,
        DurableMemoryRecord: DurableMemoryRecord.to_dict,
        MemorySelectionDecision: MemorySelectionDecision.to_dict,
        MemoryCandidate: MemoryCandidate.to_dict,
        MemoryScope: MemoryScope.to_dict,
        PolicyEffect: PolicyEffect.to_dict,
        FeedbackCalibrationSummary: FeedbackCalibrationSummary.to_dict,
        PolicyCompilation: PolicyCompilation.to_dict,
        RepositoryCacheProvenance: RepositoryCacheProvenance.to_dict,
        RepositoryKnowledgeEntry: RepositoryKnowledgeEntry.to_dict,
        MemoryCuratorDecision: MemoryCuratorDecision.to_dict,
        MemoryCandidateBatch: MemoryCandidateBatch.to_dict,
        RepositoryKnowledgeKey: RepositoryKnowledgeKey.to_dict,
        GitCommitSourceRef: GitCommitSourceRef.to_dict,
        HumanDeclarationSourceRef: HumanDeclarationSourceRef.to_dict,
        ObservationSourceRef: ObservationSourceRef.to_dict,
        RepositoryRangeSourceRef: RepositoryRangeSourceRef.to_dict,
        RepositorySymbolSourceRef: RepositorySymbolSourceRef.to_dict,
        SessionArtifactSourceRef: SessionArtifactSourceRef.to_dict,
    }
    converter = converters.get(type(value))
    if converter is not None:
        converted = converter(value)
        return dict(converted) if _is_plain_mapping(converted) else {}
    if type(value) is RepositoryCacheResult:
        return {"provenance": value.provenance}
    if type(value) is MemoryCuratorResult:
        return {"decision": value.decision, "batch": value.batch}
    wrapper_types = _feedback_wrapper_types()
    if wrapper_types and type(value) is wrapper_types[0]:
        return {"summary": value.summary}
    if len(wrapper_types) > 1 and type(value) is wrapper_types[1]:
        return {"summary": value}
    return {}


def _audit_pick(mapping: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _audit_values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if _is_plain_mapping(value) or type(value) in _canonical_audit_types():
        return (value,)
    if type(value) in {list, tuple}:
        return tuple(value[:_AUDIT_MAX_ITEMS])
    if type(value) is str:
        return (value,)
    return ()


def _audit_rows(value: Any) -> tuple[dict[str, Any], ...]:
    return tuple(
        row for item in _audit_values(value) if (row := _audit_mapping(item))
    )


def _audit_text(value: Any, *, default: str = "") -> str:
    if type(value) is str:
        return " ".join(value.split())
    if type(value) in {
        Applicability,
        CandidateStatus,
        CuratorDecisionOutcome,
        CuratorWarningCode,
        MemoryKind,
        PolicyDiagnosticCode,
        PolicyDiagnosticSeverity,
        PolicyDisposition,
        PolicyEffectKind,
        RecordStatus,
        RepositoryCacheStatus,
        RuntimeActionKind,
        SourceRefType,
        ValidityPolicy,
    }:
        return value.value
    return default


def _audit_enum(value: Any, enum_type: type[Enum]) -> str:
    text = _audit_text(value)
    allowed = {item.value for item in enum_type}
    return text if text in allowed else ""


def _audit_token(value: Any, *, default: str = "") -> str:
    text = _audit_text(value)
    return text if _AUDIT_TOKEN.fullmatch(text) else default


def _audit_bool(value: Any) -> bool | None:
    return value if type(value) is bool else None


def _audit_int(value: Any, *, minimum: int = 0) -> int | None:
    if type(value) is int and minimum <= value <= _AUDIT_MAX_INTEGER:
        return value
    return None


def _audit_digest(value: Any) -> str:
    text = _audit_text(value)
    return text if _AUDIT_DIGEST.fullmatch(text) else ""


def _audit_git(value: Any) -> str:
    text = _audit_text(value)
    return text if _AUDIT_GIT_OBJECT.fullmatch(text) else ""


def _audit_id(value: Any, prefix: str) -> str:
    text = _audit_text(value)
    return text if re.fullmatch(re.escape(prefix) + r"-[0-9a-f]{64}", text) else ""


def _audit_identifier(value: Any) -> str:
    text = _audit_token(value)
    lowered = text.casefold()
    if any(marker in lowered for marker in ("secret", "password", "token", "api_key", "credential")):
        return ""
    return text


def _audit_revision_binding(value: Any) -> str:
    text = _audit_text(value).casefold()
    if _AUDIT_GIT_OBJECT.fullmatch(text):
        return text
    if re.fullmatch(r"(?:base|head)@[0-9a-f]{40,64}", text):
        return text
    if re.fullmatch(r"[0-9a-f]{40,64}\.{2}[0-9a-f]{40,64}", text):
        return text
    return ""


def _audit_path(value: Any, *, allow_glob: bool = True) -> str:
    text = _audit_text(value)
    if not text or "\n" in text or "\r" in text or text.startswith(("/", "\\")):
        return ""
    if re.match(r"^[A-Za-z]:[/\\]", text):
        return ""
    components = text.replace("\\", "/").split("/")
    if any(component == ".." for component in components):
        return ""
    if not allow_glob and any(char in text for char in "*?[]"):
        return ""
    if any(component.casefold() in {".env", "credentials", "secrets.json", "id_rsa", "id_ed25519"} for component in components):
        return ""
    return text.replace("\\", "/")


def _audit_timestamp(value: Any) -> str:
    text = _audit_text(value)
    return text if _AUDIT_TIMESTAMP.fullmatch(text) else ""


def _audit_text_list(value: Any, *, allowed: frozenset[str] | None = None) -> list[str]:
    result: list[str] = []
    for item in _audit_values(value):
        text = _audit_text(item)
        if not text or (allowed is not None and text not in allowed):
            continue
        if text not in result:
            result.append(text)
    return result[:_AUDIT_MAX_ITEMS]


def _safe_reason_list(value: Any) -> list[str]:
    return _audit_text_list(value, allowed=_AUDIT_SAFE_REASON_CODES)


def _safe_id_list(value: Any, prefix: str) -> list[str]:
    result: list[str] = []
    for item in _audit_values(value):
        identifier = _audit_id(item, prefix)
        if identifier and identifier not in result:
            result.append(identifier)
    return result[:_AUDIT_MAX_ITEMS]


def _audit_scope(value: Any) -> dict[str, list[str]]:
    row = _audit_mapping(value)
    if not row and type(value) is str:
        row = {"paths": [value]}
    paths = [item for item in (_audit_path(v) for v in _audit_values(row.get("paths"))) if item]
    symbols = [item for item in (_audit_identifier(v) for v in _audit_values(row.get("symbols"))) if item]
    contracts = [item for item in (_audit_token(v) for v in _audit_values(row.get("contracts"))) if item]
    languages = [item for item in (_audit_token(v) for v in _audit_values(row.get("languages"))) if item]
    return {
        "paths": list(dict.fromkeys(paths))[:_AUDIT_MAX_ITEMS],
        "symbols": list(dict.fromkeys(symbols))[:_AUDIT_MAX_ITEMS],
        "contracts": list(dict.fromkeys(contracts))[:_AUDIT_MAX_ITEMS],
        "languages": list(dict.fromkeys(languages))[:_AUDIT_MAX_ITEMS],
    }


_SOURCE_FIELDS = (
    "schema_version",
    "type",
    "revision",
    "path",
    "line_start",
    "line_end",
    "content_hash",
    "qualified_name",
    "hash_kind",
    "commit_sha",
    "metadata_hash",
    "review_id",
    "observation_id",
    "revision_binding",
    "artifact_name",
    "artifact_schema",
    "artifact_hash",
    "request_id",
    "actor",
    "declaration_hash",
    "created_at",
)


def _project_source_ref(value: Any) -> dict[str, Any]:
    row = _audit_mapping(value)
    if not row:
        return {}
    source_type = _audit_enum(
        row.get("type") or row.get("source_type"), SourceRefType
    )
    if not source_type:
        return {}
    result: dict[str, Any] = {"schema_version": 1, "type": source_type}
    if source_type in {"repository_range", "repository_symbol"}:
        revision = _audit_git(row.get("revision"))
        path = _audit_path(row.get("path"), allow_glob=False)
        if revision:
            result["revision"] = revision
        if path:
            result["path"] = path
        if source_type == "repository_range":
            start = _audit_int(row.get("line_start"), minimum=1)
            end = _audit_int(row.get("line_end"), minimum=1)
            if start is not None and end is not None and end >= start:
                result["line_start"] = start
                result["line_end"] = end
        else:
            qualified_name = _audit_identifier(row.get("qualified_name"))
            hash_kind = _audit_token(row.get("hash_kind"))
            if qualified_name:
                result["qualified_name"] = qualified_name
            if hash_kind in {"signature", "body"}:
                result["hash_kind"] = hash_kind
        content_hash = _audit_digest(row.get("content_hash"))
        if content_hash:
            result["content_hash"] = content_hash
    elif source_type == "git_commit":
        commit_sha = _audit_git(row.get("commit_sha"))
        if commit_sha:
            result["commit_sha"] = commit_sha
        metadata_hash = _audit_digest(row.get("metadata_hash"))
        if metadata_hash:
            result["metadata_hash"] = metadata_hash
    elif source_type == "observation":
        review_id = _audit_identifier(row.get("review_id"))
        observation_id = _audit_text(row.get("observation_id"))
        binding = _audit_revision_binding(row.get("revision_binding"))
        content_hash = _audit_digest(row.get("content_hash"))
        if review_id:
            result["review_id"] = review_id
        if _AUDIT_OBSERVATION_ID.fullmatch(observation_id):
            result["observation_id"] = observation_id
        if binding:
            result["revision_binding"] = binding
        if content_hash:
            result["content_hash"] = content_hash
    elif source_type == "session_artifact":
        for key in ("review_id", "artifact_name", "artifact_schema"):
            text = _audit_identifier(row.get(key))
            if text:
                result[key] = text
        binding = _audit_revision_binding(row.get("revision_binding"))
        artifact_hash = _audit_digest(row.get("artifact_hash"))
        if binding:
            result["revision_binding"] = binding
        if artifact_hash:
            result["artifact_hash"] = artifact_hash
    elif source_type == "human_declaration":
        request_id = _audit_id(row.get("request_id"), "REQ")
        actor = _audit_identifier(row.get("actor"))
        declaration_hash = _audit_digest(row.get("declaration_hash"))
        created_at = _audit_timestamp(row.get("created_at"))
        if request_id:
            result["request_id"] = request_id
        if actor:
            result["actor"] = actor
        if declaration_hash:
            result["declaration_hash"] = declaration_hash
        if created_at:
            result["created_at"] = created_at
    return result if len(result) > 2 else {}


def _project_policy_effect(value: Any) -> dict[str, Any] | None:
    row = _audit_mapping(value)
    if not row:
        return None
    effect_type = _audit_enum(
        row.get("type") or row.get("effect_kind"), PolicyEffectKind
    )
    value_text = _audit_token(row.get("value"))
    if not effect_type or not value_text:
        return None
    if effect_type == PolicyEffectKind.RISK_FLOOR.value and value_text not in {
        "low",
        "medium",
        "high",
        "critical",
    }:
        return None
    return {"schema_version": 1, "type": effect_type, "value": value_text}


def _project_validity(
    record: Mapping[str, Any], decision: Mapping[str, Any] | None
) -> dict[str, Any]:
    source = _audit_mapping(record.get("validity"))
    selected = decision or source
    result: dict[str, Any] = {}
    applicability = _audit_enum(selected.get("applicability"), Applicability)
    if applicability:
        result["applicability"] = applicability
    status = _audit_enum(record.get("status"), RecordStatus)
    if status:
        result["status"] = status
    valid_from = _audit_git(record.get("valid_from_sha") or source.get("valid_from_sha"))
    if valid_from:
        result["valid_from_sha"] = valid_from
    target_head = _audit_git(source.get("target_head"))
    if target_head:
        result["target_head"] = target_head
    policies = source.get("policies") or source.get("validity_policies")
    if policies is None:
        policies = record.get("validity_policies")
    safe_policies = _audit_text_list(
        policies,
        allowed=frozenset(item.value for item in ValidityPolicy),
    )
    if safe_policies:
        result["policies"] = safe_policies
    reasons = selected.get("reason_codes")
    safe_reasons = _safe_reason_list(reasons)
    if safe_reasons:
        result["reason_codes"] = safe_reasons
    requires = _audit_bool(
        selected.get("requires_revalidation")
        if "requires_revalidation" in selected
        else source.get("requires_revalidation")
    )
    if requires is not None:
        result["requires_revalidation"] = requires
    return result


def _authority_failure(
    record: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
    compiled_provenance: Mapping[str, Any] | None,
) -> str | None:
    """Return a fixed reason unless the complete authority chain is present."""

    memory_id = _audit_id(record.get("memory_id") or record.get("id"), "MEM")
    if not memory_id:
        return "invalid_record"
    raw_status = _audit_text(record.get("status"))
    status_value = _audit_enum(record.get("status"), RecordStatus)
    if status_value == "":
        return "record_status_missing" if not raw_status else "invalid_record"
    if status_value != RecordStatus.ACTIVE.value:
        return "invalid_record"
    active_flag = record.get("active")
    if active_flag is not None and active_flag is not True:
        return "invalid_record"
    kind = _audit_enum(record.get("kind"), MemoryKind)
    if not kind:
        return "invalid_record"
    selection = decision
    if selection is None:
        selection = _audit_mapping(record.get("selection_decision"))
    if not selection:
        return "selection_missing"
    applicability = _audit_enum(selection.get("applicability"), Applicability)
    if not applicability:
        return "selection_missing"
    if applicability != Applicability.SELECTED.value:
        return "selection_not_selected"
    candidate_id = _audit_id(record.get("candidate_id"), "MC")
    bundle_hash = _audit_digest(record.get("source_bundle_hash"))
    approved_by = _audit_identifier(record.get("approved_by"))
    approval_event_id = _audit_id(record.get("approval_event_id"), "EVT")
    if not candidate_id or not bundle_hash or not approved_by or not approval_event_id:
        return "approval_missing"
    if not compiled_provenance:
        return "compiled_provenance_missing"
    if compiled_provenance.get("disposition") not in {
        PolicyDisposition.APPLIED.value,
        PolicyDisposition.INFORMATIONAL.value,
    }:
        return "compiled_provenance_not_authoritative"
    return None


def _project_applied_memory(
    record: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
    compiled_provenance: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    memory_id = _audit_id(record.get("memory_id") or record.get("id"), "MEM")
    if (
        not memory_id
        or _authority_failure(record, decision, compiled_provenance) is not None
    ):
        return None
    kind = _audit_enum(record.get("kind"), MemoryKind) or "unknown"
    candidate_id = _audit_id(record.get("candidate_id"), "MC")
    bundle_hash = _audit_digest(record.get("source_bundle_hash"))
    approved_by = _audit_identifier(record.get("approved_by"))
    approval_event_id = _audit_id(record.get("approval_event_id"), "EVT")
    result: dict[str, Any] = {
        "memory_id": memory_id,
        "kind": kind,
        "scope": _audit_scope(record.get("scope")),
        "authority": (
            "runtime_compiled_policy"
            if compiled_provenance.get("disposition")
            == PolicyDisposition.APPLIED.value
            else "human_approved_context"
        ),
        "status": RecordStatus.ACTIVE.value,
        "source_refs": [
            projected
            for projected in (
                _project_source_ref(item)
                for item in _audit_values(record.get("source_refs"))
            )
            if projected
        ][:_AUDIT_MAX_ITEMS],
        "validity": _project_validity(record, decision),
        "selection_decision": {
            "applicability": Applicability.SELECTED.value,
            "reason_codes": _safe_reason_list(
                (decision or _audit_mapping(record.get("selection_decision"))).get(
                    "reason_codes"
                )
            ),
        },
    }
    statement = _audit_text(record.get("statement"))
    if statement:
        result["statement"] = statement
    result.update(
        {
            "candidate_id": candidate_id,
            "approved_by": approved_by,
            "approval_event_id": approval_event_id,
            "source_bundle_hash": bundle_hash,
        }
    )
    effect = _project_policy_effect(record.get("policy_effect"))
    if effect is not None:
        result["policy_effect"] = effect
    return result


def _project_not_applied_memory(
    record: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
    reason: str,
) -> dict[str, Any] | None:
    memory_id = _audit_id(record.get("memory_id") or record.get("id"), "MEM")
    if not memory_id:
        return None
    validity = _audit_mapping(record.get("validity"))
    selected = decision or validity
    applicability = _audit_enum(
        selected.get("applicability") if selected else None,
        Applicability,
    ) or "missing"
    status = _audit_enum(record.get("status"), RecordStatus) or "missing"
    kind = _audit_enum(record.get("kind"), MemoryKind) or "unknown"
    claimed_authority = _audit_text(record.get("authority"))
    if claimed_authority not in {
        "human_approved_context",
        "runtime_compiled_policy",
    }:
        claimed_authority = ""
    result: dict[str, Any] = {
        "memory_id": memory_id,
        "kind": kind,
        "scope": _audit_scope(record.get("scope")),
        "status": status,
        "applicability": applicability,
        "reason_code": reason if reason in _AUDIT_DEGRADATION_CODES else "record_not_applied",
        "authority": "not_applied",
    }
    if claimed_authority:
        result["claimed_authority"] = claimed_authority
    return result


def _project_not_applied_rows(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in _audit_values(value):
        row = _audit_mapping(item)
        if not row:
            continue
        memory_id = _audit_id(row.get("memory_id"), "MEM")
        if not memory_id:
            continue
        projected = {
            "memory_id": memory_id,
            "kind": _audit_enum(row.get("kind"), MemoryKind) or "unknown",
            "scope": _audit_scope(row.get("scope")),
            "status": _audit_enum(row.get("status"), RecordStatus) or "missing",
            "applicability": _audit_enum(row.get("applicability"), Applicability) or "missing",
            "reason_code": _audit_token(row.get("reason_code"), default="record_not_applied"),
            "authority": "not_applied",
        }
        claimed = _audit_text(row.get("claimed_authority"))
        if claimed in {"human_approved_context", "runtime_compiled_policy"}:
            projected["claimed_authority"] = claimed
        result.append(projected)
    return result[:_AUDIT_MAX_ITEMS]


def _project_compiled_policy(value: Any) -> dict[str, Any]:
    row = _audit_mapping(value)
    if not row:
        return {}
    result: dict[str, Any] = {}
    if _audit_text(row.get("policy_version")) == "memory_policy_v1":
        result["policy_version"] = "memory_policy_v1"
    for key in ("initial_risk_floor", "effective_risk_floor"):
        level = _audit_token(row.get(key))
        if level in {"low", "medium", "high", "critical"}:
            result[key] = level
    actions: list[dict[str, Any]] = []
    for item in _audit_values(row.get("actions")):
        action = _audit_mapping(item)
        if not action:
            continue
        action_type = _audit_enum(action.get("type"), RuntimeActionKind)
        if not action_type:
            continue
        projected: dict[str, Any] = {}
        projected["type"] = action_type
        if action_type == RuntimeActionKind.RAISE_RISK_FLOOR.value:
            level = _audit_token(action.get("minimum_level"))
            if level not in {"low", "medium", "high", "critical"}:
                continue
            projected["minimum_level"] = level
        elif action_type == RuntimeActionKind.REQUIRE_CONTRACT.value:
            identifier = _audit_identifier(action.get("contract_id"))
            if not identifier:
                continue
            projected["contract_id"] = identifier
        elif action_type == RuntimeActionKind.REQUIRE_CHECK.value:
            identifier = _audit_identifier(action.get("check_id"))
            if not identifier:
                continue
            projected["check_id"] = identifier
        else:
            identifier = _audit_identifier(action.get("command_template_id"))
            if not identifier:
                continue
            projected["command_template_id"] = identifier
        memory_ids = _safe_id_list(action.get("memory_ids"), "MEM")
        if memory_ids:
            projected["memory_ids"] = memory_ids
        if projected:
            actions.append(projected)
    if actions:
        result["actions"] = actions[:_AUDIT_MAX_ITEMS]
    diagnostics: list[dict[str, Any]] = []
    for item in _audit_values(row.get("diagnostics")):
        diagnostic = _audit_mapping(item)
        if not diagnostic:
            continue
        code = _audit_enum(diagnostic.get("code"), PolicyDiagnosticCode)
        severity = _audit_enum(diagnostic.get("severity"), PolicyDiagnosticSeverity)
        if not code or not severity:
            continue
        projected: dict[str, Any] = {"code": code, "severity": severity}
        memory_id = _audit_id(diagnostic.get("memory_id"), "MEM")
        if memory_id:
            projected["memory_id"] = memory_id
        if projected:
            diagnostics.append(projected)
    if diagnostics:
        result["diagnostics"] = diagnostics[:_AUDIT_MAX_ITEMS]
    provenance: list[dict[str, Any]] = []
    for item in _audit_values(row.get("provenance")):
        source = _audit_mapping(item)
        if not source:
            continue
        memory_id = _audit_id(source.get("memory_id"), "MEM")
        disposition = _audit_enum(source.get("disposition"), PolicyDisposition)
        if not memory_id or not disposition:
            continue
        projected = {"memory_id": memory_id, "disposition": disposition}
        for key, prefix in (("candidate_id", "MC"), ("approval_event_id", "EVT")):
            identifier = _audit_id(source.get(key), prefix)
            if identifier:
                projected[key] = identifier
        approved_by = _audit_identifier(source.get("approved_by"))
        if approved_by:
            projected["approved_by"] = approved_by
        effect_kind = _audit_enum(source.get("effect_kind"), PolicyEffectKind)
        runtime_kind = _audit_enum(source.get("runtime_action_kind"), RuntimeActionKind)
        expected_runtime = {
            PolicyEffectKind.RISK_FLOOR.value: RuntimeActionKind.RAISE_RISK_FLOOR.value,
            PolicyEffectKind.REQUIRE_CONTRACT.value: RuntimeActionKind.REQUIRE_CONTRACT.value,
            PolicyEffectKind.REQUIRE_CHECK.value: RuntimeActionKind.REQUIRE_CHECK.value,
            PolicyEffectKind.VERIFICATION_HINT.value: RuntimeActionKind.VERIFICATION_HINT.value,
        }
        effect_value = _audit_token(source.get("effect_value"))
        if disposition == PolicyDisposition.APPLIED.value and (
            not effect_kind
            or not effect_value
            or runtime_kind != expected_runtime.get(effect_kind)
        ):
            continue
        if disposition == PolicyDisposition.INFORMATIONAL.value and (
            effect_kind or effect_value or runtime_kind
        ):
            continue
        if effect_kind:
            projected["effect_kind"] = effect_kind
        if runtime_kind:
            projected["runtime_action_kind"] = runtime_kind
        if effect_value:
            projected["effect_value"] = effect_value
        diagnostic_codes = _audit_text_list(
            source.get("diagnostic_codes"),
            allowed=frozenset(item.value for item in PolicyDiagnosticCode),
        )
        if diagnostic_codes:
            projected["diagnostic_codes"] = diagnostic_codes
        if projected:
            provenance.append(projected)
    if provenance:
        result["provenance"] = sorted(provenance, key=lambda item: item["memory_id"])
    blocked = _audit_bool(row.get("blocked"))
    result["blocked"] = bool(
        blocked is True
        or any(item.get("severity") == "blocking" for item in diagnostics)
    )
    return result


def _project_cache_provenance(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if type(value) is RepositoryCacheResult:
        value = value.provenance
    row = _audit_mapping(value)
    if "provenance" in row and not row.get("status"):
        value = row["provenance"]
    elif "entries" in row:
        value = row["entries"]
    rows: list[dict[str, Any]] = []
    for item in _audit_values(value):
        source = _audit_mapping(item)
        if type(item) is RepositoryKnowledgeEntry and source:
            source["status"] = "referenced"
        if "key" in source:
            key = _audit_mapping(source.get("key"))
            for name in (
                "key_hash",
                "repository_key",
                "revision_binding",
                "capability",
                "configuration_digest",
                "input_digest",
            ):
                if name not in source and name in key:
                    source[name] = key[name]
            if "analyzer" not in source:
                source["analyzer"] = {
                    name: key[name]
                    for name in ("analyzer_name", "analyzer_version")
                    if name in key
                }
        if not source and type(item) is str:
            source = {"entry_id": item, "status": "referenced"}
        projected: dict[str, Any] = {}
        status = _audit_enum(source.get("status"), RepositoryCacheStatus)
        if not status and source.get("status") == "referenced":
            status = "referenced"
        if status:
            projected["status"] = status
        for name in ("key_hash", "configuration_digest", "input_digest", "blob_hash", "summary_hash"):
            digest = _audit_digest(source.get(name))
            if digest:
                projected[name] = digest
        repository_key = _audit_digest(source.get("repository_key"))
        if repository_key:
            projected["repository_key"] = repository_key
        binding = _audit_revision_binding(source.get("revision_binding"))
        if binding:
            projected["revision_binding"] = binding
        capability = _audit_token(source.get("capability"))
        if capability in {
            "file_index", "symbol_index", "definitions", "references", "calls", "tests", "project_config", "git_summary"
        }:
            projected["capability"] = capability
        entry_id = _audit_id(source.get("entry_id"), "RKE")
        if entry_id:
            projected["entry_id"] = entry_id
        if type(source.get("persistent")) is bool:
            projected["persistent"] = source["persistent"]
        if type(source.get("session_pinned")) is bool:
            projected["session_pinned"] = source["session_pinned"]
        artifact_schema = _audit_token(source.get("artifact_schema"))
        if artifact_schema:
            projected["artifact_schema"] = artifact_schema
        size_bytes = _audit_int(source.get("size_bytes"))
        if size_bytes is not None:
            projected["size_bytes"] = size_bytes
        corruption = _audit_token(source.get("corruption_reason"))
        if corruption:
            projected["corruption_reason"] = (
                corruption if corruption in _AUDIT_SAFE_CACHE_CORRUPTION else "unknown"
            )
        analyzer = _audit_mapping(source.get("analyzer"))
        if analyzer:
            analyzer_result: dict[str, str] = {}
            for key in ("name", "version", "analyzer_name", "analyzer_version"):
                text = _audit_token(analyzer.get(key))
                if text:
                    analyzer_result[key] = text
            if analyzer_result:
                projected["analyzer"] = analyzer_result
        fallback = _audit_mapping(source.get("fallback"))
        if fallback:
            safe_fallback: dict[str, str] = {}
            for key, item_value in fallback.items():
                if key not in _AUDIT_SAFE_FALLBACK_KEYS:
                    continue
                safe_value = _audit_token(item_value)
                if safe_value in _AUDIT_SAFE_FALLBACK_VALUES:
                    safe_fallback[key] = safe_value
            if safe_fallback:
                projected["fallback"] = safe_fallback
        if projected:
            rows.append(projected)
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in rows:
        key = (
            item.get("entry_id", ""),
            item.get("status", ""),
            item.get("key_hash", ""),
        )
        unique[key] = item
    return list(unique.values())[:_AUDIT_MAX_ITEMS]


def _project_warnings(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in _audit_values(value):
        row = _audit_mapping(item)
        if not row:
            text = _audit_enum(item, Applicability)
            category = _warning_category(text, ())
            if category:
                result.append({"category": category, "code": text})
            continue
        applicability = _audit_enum(
            row.get("applicability") or row.get("category"), Applicability
        )
        category = _warning_category(applicability, row.get("reason_codes", []))
        if category is None:
            category = _audit_token(row.get("category"))
            if category not in {"lineage", "revalidation", "stale", "omitted"}:
                continue
        projected: dict[str, Any] = {"category": category}
        memory_id = _audit_id(row.get("memory_id"), "MEM")
        if memory_id:
            projected["memory_id"] = memory_id
        if applicability:
            projected["applicability"] = applicability
        code = _audit_token(row.get("code"))
        if code in _AUDIT_SAFE_REASON_CODES:
            projected["code"] = code
        safe_reasons = _safe_reason_list(row.get("reason_codes"))
        if safe_reasons:
            projected["reason_codes"] = safe_reasons
        requires = _audit_bool(row.get("requires_revalidation"))
        if requires is not None:
            projected["requires_revalidation"] = requires
        result.append(projected)
    return result


def _warning_category(applicability: str, reasons: Any) -> str | None:
    text = " ".join([applicability, *_safe_reason_list(reasons)]).casefold()
    if "lineage" in text or applicability in {"lineage_mismatch", "diverged"}:
        return "lineage"
    if "revalid" in text or applicability in {"source_changed", "source_missing"}:
        return "revalidation"
    if applicability in {"not_yet_valid", "expired", "revoked", "superseded"}:
        return "stale"
    if applicability == "budget_omitted":
        return "omitted"
    return None


def _unique_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            _audit_text(row.get("category")),
            _audit_text(row.get("memory_id")),
            _audit_text(row.get("applicability") or row.get("code")),
            ",".join(_safe_reason_list(row.get("reason_codes"))),
        )
        unique[key] = row
    return [unique[key] for key in sorted(unique)][:_AUDIT_MAX_ITEMS]


def _project_feedback_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    wrapper_types = _feedback_wrapper_types()
    if wrapper_types and type(value) is wrapper_types[0]:
        value = value.summary
    elif len(wrapper_types) > 1 and type(value) is wrapper_types[1]:
        value = wrapper_types[1].to_dict(value)
    row = _audit_mapping(value)
    if "summary" in row:
        row = _audit_mapping(row.get("summary"))
    if not row:
        return {}
    feedback_ids = row.get("source_feedback_ids") or row.get("feedback_ids") or []
    review_ids = row.get("source_review_ids") or row.get("review_ids") or []
    result: dict[str, Any] = {}
    version = _audit_token(row.get("policy_version") or row.get("version"))
    if version == "feedback_aggregation_v1":
        result["policy_version"] = version
    sample_count = _audit_int(row.get("sample_count"))
    if sample_count is None and feedback_ids:
        sample_count = len(_safe_id_list(feedback_ids, "FB"))
    if sample_count is not None:
        result["sample_count"] = sample_count
    review_count = _audit_int(row.get("review_count"))
    if review_count is None and review_ids:
        review_count = len(
            {
                review_id
                for item in _audit_values(review_ids)
                if (review_id := _audit_identifier(item))
            }
        )
    if review_count is not None:
        result["review_count"] = review_count
    eligible = _audit_bool(row.get("eligible"))
    if eligible is not None:
        result["eligible"] = eligible
    return result


def _project_pending_candidates(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    row = _audit_mapping(value)
    if "batch" in row:
        value = row["batch"]
        row = _audit_mapping(value)
    if "candidates" in row:
        value = row["candidates"]
    result: list[dict[str, Any]] = []
    for item in _audit_values(value):
        source = _audit_mapping(item)
        candidate_id = _audit_id(
            source.get("candidate_id") or source.get("id"), "MC"
        )
        if not candidate_id:
            continue
        status = _audit_enum(source.get("status"), CandidateStatus)
        if status != CandidateStatus.PENDING_APPROVAL.value:
            continue
        sensitivity = _audit_token(source.get("sensitivity"))
        projected: dict[str, Any] = {
            "candidate_id": candidate_id,
            "kind": _audit_enum(source.get("kind"), MemoryKind) or "unknown",
            "scope": _audit_scope(source.get("scope")),
            "status": status,
            "active": False,
            "approval_hint": (
                f"memory approve {candidate_id} --actor <actor> --reason <reason>"
            ),
        }
        statement = _audit_text(source.get("statement"))
        if statement and sensitivity != "blocked":
            projected["statement"] = statement
        result.append(projected)
    return sorted(result, key=lambda item: item["candidate_id"])[:_AUDIT_MAX_ITEMS]


def _project_candidate_outcomes(value: Any) -> list[dict[str, Any]]:
    """Project lifecycle outcomes without exposing Candidate content."""

    outcomes: dict[str, dict[str, Any]] = {}
    for item in _audit_values(value):
        source = _audit_mapping(item)
        candidate_id = _audit_id(source.get("candidate_id"), "MC")
        status = _audit_enum(source.get("status"), CandidateStatus)
        dedupe = _audit_token(source.get("dedupe"))
        replayed = _audit_bool(source.get("replayed"))
        persistence = _audit_token(source.get("persistence"))
        validation_hash = _audit_digest(
            source.get("validation_report_hash")
        )
        if (
            not candidate_id
            or not status
            or dedupe not in _AUDIT_SAFE_CANDIDATE_DEDUPE
            or replayed is None
            or persistence not in _AUDIT_SAFE_CANDIDATE_PERSISTENCE
            or (replayed and persistence != "replayed")
            or (not replayed and persistence != "persisted")
            or not validation_hash
        ):
            continue
        outcomes[candidate_id] = {
            "candidate_id": candidate_id,
            "status": status,
            "dedupe": dedupe,
            "replayed": replayed,
            "persistence": persistence,
            "validation_report_hash": validation_hash,
        }
    return [outcomes[key] for key in sorted(outcomes)][:_AUDIT_MAX_ITEMS]


def _project_outbox(value: Any) -> dict[str, Any]:
    row = _audit_mapping(value)
    if not row:
        return {}
    result: dict[str, Any] = {}
    status = _audit_token(row.get("status"))
    if status in _AUDIT_SAFE_OUTBOX_STATUSES:
        result["status"] = status
    request_id = _audit_id(row.get("request_id"), "REQ")
    if request_id:
        result["request_id"] = request_id
    request_hash = _audit_digest(row.get("request_hash"))
    if request_hash:
        result["request_hash"] = request_hash
    review_id = _audit_identifier(row.get("review_id"))
    if review_id:
        result["review_id"] = review_id
    receipt_id = _audit_text(row.get("receipt_id"))
    if _AUDIT_STABLE_ID.fullmatch(receipt_id):
        result["receipt_id"] = receipt_id
    error_code = _audit_token(row.get("error_code"))
    if error_code in _AUDIT_SAFE_OUTBOX_CODES:
        result["error_code"] = error_code
    if row.get("candidate_ids") is not None:
        result["candidate_ids"] = _safe_id_list(row["candidate_ids"], "MC")
    if row.get("candidate_outcomes") is not None:
        result["candidate_outcomes"] = _project_candidate_outcomes(
            row["candidate_outcomes"]
        )
    pending = _audit_bool(row.get("pending"))
    if pending is not None:
        result["pending"] = pending
    elif status in {"pending", "outbox_pending"}:
        result["pending"] = True
    if result.get("pending") and result.get("review_id"):
        result["replay_hint"] = f"memory replay-outbox {result['review_id']}"
    return result


def _project_curator(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if type(value) is MemoryCuratorResult:
        value = value.decision
    row = _audit_mapping(value)
    if "decision" in row:
        row = _audit_mapping(row.get("decision"))
    if not row:
        return {}
    result: dict[str, Any] = {}
    mode = _audit_token(row.get("mode"))
    if mode in {"local", "model", "disabled", "off"}:
        result["mode"] = mode
    else:
        mode = ""
    outcome = _audit_enum(row.get("outcome"), CuratorDecisionOutcome)
    if outcome:
        result["outcome"] = outcome
    status = _audit_token(row.get("status"))
    if status not in {
        "proposed",
        "empty",
        "rejected",
        "fallback",
        "disabled",
        "skipped",
        "failed",
    }:
        status = ""
    if row.get("disabled") is True or mode in {"disabled", "off"}:
        status = "disabled"
    elif not status:
        status = "fallback" if outcome == "rejected" and row.get("warning_codes") else outcome
    if status:
        result["status"] = status
    attempt_count = _audit_int(row.get("attempt_count"))
    if attempt_count is not None:
        result["attempt_count"] = attempt_count
    impact = _audit_token(row.get("review_conclusion_impact"))
    if impact in _AUDIT_SAFE_REVIEW_IMPACTS:
        result["review_conclusion_impact"] = impact
    if row.get("candidate_ids") is not None:
        result["candidate_ids"] = _safe_id_list(row["candidate_ids"], "MC")
    if row.get("warning_codes") is not None:
        result["warning_codes"] = _audit_text_list(
            row["warning_codes"],
            allowed=frozenset(item.value for item in CuratorWarningCode),
        )
    return result


def _project_memory_status(
    value: Any,
    *,
    policy: Mapping[str, Any],
    curator: Mapping[str, Any],
    outbox: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is str and value in _AUDIT_DEGRADATION_CODES:
        row = {"available": False, "unavailable_reason": value}
    else:
        row = _audit_mapping(value)
    result: dict[str, Any] = {}
    mode = _audit_token(row.get("mode"))
    if mode in {"off", "read", "read-write"}:
        result["mode"] = mode
    safe_statuses = {
        "disabled", "skipped", "empty", "selected", "completed", "partial", "failed", "unavailable", "hit", "miss", "rebuild", "off", "pending", "persisted"
    }
    for key in ("selection_status", "proposal_status", "store_status", "cache_status"):
        text = _audit_token(row.get(key))
        if text in safe_statuses:
            result[key] = text
    required = _audit_bool(row.get("required"))
    if required is not None:
        result["required"] = required
    available = _audit_bool(row.get("available"))
    unavailable = _audit_bool(row.get("memory_unavailable"))
    unavailable_detail = _audit_mapping(row.get("unavailable"))
    if unavailable is None and unavailable_detail:
        unavailable = True
    if unavailable is None:
        unavailable = _audit_bool(row.get("unavailable"))
    if unavailable is None:
        unavailable = available is False
    result["memory_unavailable"] = unavailable is True
    if available is not None:
        result["available"] = available
    reason = _audit_token(
        row.get("unavailable_reason")
        or row.get("reason_code")
        or unavailable_detail.get("reason_code")
        or unavailable_detail.get("reason")
    )
    if reason not in _AUDIT_DEGRADATION_CODES and reason not in _AUDIT_SAFE_CACHE_CORRUPTION:
        reason = "unknown" if reason else ""
    if reason:
        result["unavailable_reason"] = reason
    hard_blocked = _audit_bool(row.get("hard_policy_blocked"))
    hard_detail = _audit_mapping(row.get("hard_policy"))
    if hard_blocked is not True and hard_detail:
        hard_blocked = _audit_bool(
            hard_detail.get("blocked")
            if "blocked" in hard_detail
            else hard_detail.get("hard_policy_blocked")
        )
    result["hard_policy_blocked"] = (
        hard_blocked is True or policy.get("blocked") is True
    )
    result["outbox_pending"] = (
        row.get("outbox_pending") is True
        or outbox.get("pending") is True
        or outbox.get("status") in {"pending", "outbox_pending"}
    )
    if outbox:
        result["outbox"] = dict(outbox)
    if curator:
        result["curator"] = dict(curator)
    reasons = _audit_text_list(
        row.get("degradation_reasons"),
        allowed=_AUDIT_DEGRADATION_CODES | _AUDIT_LIMIT_CODES,
    )
    if result["memory_unavailable"] and "memory_unavailable" not in reasons:
        reasons.append("memory_unavailable")
    if reason in _AUDIT_DEGRADATION_CODES and reason not in reasons:
        reasons.append(reason)
    if result["hard_policy_blocked"] and "hard_policy_blocked" not in reasons:
        reasons.append("hard_policy_blocked")
    if result["outbox_pending"] and "outbox_pending" not in reasons:
        reasons.append("outbox_pending")
    if curator.get("status") == "fallback" and "curator_fallback" not in reasons:
        reasons.append("curator_fallback")
    if reasons:
        result["degraded"] = True
        result["degradation_reasons"] = reasons
    else:
        degraded = _audit_bool(row.get("degraded"))
        if degraded is not None:
            result["degraded"] = degraded
    return result


def _project_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    result: dict[str, Any] = {}
    snapshot_id = _audit_id(row.get("snapshot_id"), "MSNAP")
    snapshot_hash = _audit_digest(row.get("snapshot_hash"))
    if snapshot_id:
        result["snapshot_id"] = snapshot_id
    if snapshot_hash:
        result["snapshot_hash"] = snapshot_hash
    for key in ("base_sha", "head_sha"):
        revision = _audit_git(row.get(key))
        if revision:
            result[key] = revision
    selection_policy = _audit_text(row.get("selection_policy_version"))
    if selection_policy in SUPPORTED_MEMORY_SELECTION_POLICY_VERSIONS:
        result["selection_policy_version"] = selection_policy
    for key in (
        "store_schema_version",
        "memory_generation",
        "feedback_generation",
        "knowledge_generation",
    ):
        number = _audit_int(row.get(key))
        if number is not None:
            result[key] = number
    generations = _audit_mapping(row.get("generations"))
    for key in ("store_schema_version", "memory_generation", "feedback_generation", "knowledge_generation"):
        if key not in result and key in generations:
            number = _audit_int(generations[key])
            if number is not None:
                result[key] = number
    refs = _safe_id_list(row.get("repository_knowledge_refs"), "RKE")
    if refs:
        result["repository_knowledge_refs"] = refs
    return result


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
