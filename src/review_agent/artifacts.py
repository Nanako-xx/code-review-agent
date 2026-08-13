from __future__ import annotations

import re
from types import MappingProxyType


ARTIFACT_SCHEMAS = {
    "request": "review_request_v1",
    "change_summary": "change_summary_v1",
    "intent_candidates": "intent_candidates_v1",
    "intent_questions": "intent_questions_v1",
    "intent_inference": "intent_inference_run_v1",
    "intent_observations": "observation_log_jsonl_v1",
    "intent_resolution_request": "intent_resolution_request_v1",
    "intent_events": "intent_events_v1",
    "intent": "intent_packet_v2",
    "risk_packet": "risk_packet_v1",
    "risk": "risk_assessment_v1",
    "risk_model_envelope": "risk_model_envelope_v1",
    "risk_model_raw_response": "risk_model_raw_response_v1",
    "risk_model_decision": "risk_model_decision_v1",
    "portfolio_packet": "portfolio_packet_v1",
    "portfolio_model_envelope": "portfolio_model_envelope_v1",
    "portfolio_model_raw_response": "portfolio_model_raw_response_v1",
    "portfolio_model_decision": "portfolio_model_decision_v1",
    "portfolio_plan": "portfolio_plan_v1",
    "planning_summary": "planning_summary_v1",
    "assignments": "reviewer_assignments_v1",
    "quality_gate_plan": "quality_gate_plan_v1",
    "quality_gates": "quality_gate_results_v1",
    "quality_gate_observations": "observation_log_jsonl_v1",
    "deep_quality_gates": "quality_gate_results_v1",
    "deep_quality_gate_observations": "observation_log_jsonl_v1",
    "incremental_priority": "incremental_priority_map_v1",
    "repository_intelligence": "repository_intelligence_v1",
    "repository_observations": "observation_log_jsonl_v1",
    "memory_selection_input": "memory_selection_input_v1",
    "memory_snapshot": "memory_snapshot_v1",
    "memory_selection_decision": "memory_selection_decision_v1",
    "memory_feedback_summary": "feedback_calibration_summary_v1",
    "memory_curator_envelope": "memory_curator_envelope_v1",
    "memory_curator_raw_response": "memory_curator_raw_response_v1",
    "memory_curator_decision": "memory_curator_decision_v1",
    "memory_candidates": "memory_candidate_batch_v1",
    "memory_outbox": "memory_candidate_outbox_v1",
    "memory_persistence_receipt": "memory_persistence_receipt_v1",
    "multi_reviewer": "multi_reviewer_result_v1",
    "reviewer_envelope": "model_request_envelope_v1",
    "reviewer_raw_response": "model_raw_response_v1",
    "reviewer": "reviewer_result_v1",
    "reviewer_agent_trace": "reviewer_agent_trace_v1",
    "reconciliation_prepass": "reconciliation_prepass_v1",
    "reconciliation_packet": "reconciliation_packet_v1",
    "supplemental_initial_plan": "supplemental_plan_v1",
    "reconciliation_analysis_summary": "reconciliation_analysis_summary_v1",
    "semantic_reconciliation": "semantic_reconciliation_v1",
    "reconciliation": "evidence_reconciliation_v1",
    "supplemental_summary": "supplemental_summary_v1",
    "completion": "completion_check_v1",
    "final_risk": "final_risk_assessment_v1",
    "review_brief": "review_brief_v1",
    "report": "review_report_markdown_v1",
    "observations": "observation_log_jsonl_v1",
    # Session v6 / PRWorkspace artifacts use distinct logical names so the
    # v1-v5 readers above retain their original schema bindings.
    "pr_workspace_manifest": "pr_workspace_manifest_v1",
    "snapshot_manifest": "snapshot_manifest_v1",
    "diff_artifact_index": "diff_artifact_index_v1",
    "preflight_result": "preflight_result_v1",
    "quality_gate_v2": "quality_gate_result_v2",
    "changed_symbols_v2": "changed_symbols_v2",
    "intent_packet": "intent_packet_v2_minimal",
    "intent_version": "intent_version_envelope_v1",
    "intent_analysis_record": "intent_analysis_record_v2",
    "risk_decision": "risk_decision_v2",
    "review_plan": "review_plan_v2",
    "reviewer_assignment": "reviewer_assignment_v2",
    "reviewer_output": "reviewer_output_v2",
    "aggregation_record": "aggregation_record_v1",
    "review_result": "review_result_v1",
    "context_manifest": "context_manifest_v1",
    "execution_journal_event": "execution_journal_event_v1",
}

# This contract intentionally contains only stable wire strings so SessionStore
# can enforce phase ownership without importing Session or RunPhase here.
MEMORY_ARTIFACT_PHASES = MappingProxyType(
    {
        "memory_selection_input": "memory_selection",
        "memory_snapshot": "memory_selection",
        "memory_selection_decision": "memory_selection",
        "memory_feedback_summary": "memory_selection",
        "memory_curator_envelope": "memory_proposal",
        "memory_curator_raw_response": "memory_proposal",
        "memory_curator_decision": "memory_proposal",
        "memory_candidates": "memory_proposal",
        "memory_outbox": "memory_proposal",
        "memory_persistence_receipt": "memory_proposal",
    }
)
MEMORY_ARTIFACT_SCHEMAS = MappingProxyType(
    {
        name: ARTIFACT_SCHEMAS[name]
        for name in MEMORY_ARTIFACT_PHASES
    }
)

SESSION_V6_ARTIFACT_PHASES = MappingProxyType(
    {
        "preflight.request": "preflight",
        "preflight.diff_patch": "preflight",
        "preflight.diff_index": "preflight",
        "preflight.quality_gate": "preflight",
        "preflight.changed_symbols": "preflight",
        "intent.packet": "intent",
        "planning.risk": "planning",
        "planning.review_plan": "planning",
        "aggregation.record": "aggregation",
        "aggregation.review_result": "aggregation",
        "aggregation.review_markdown": "aggregation",
    }
)

_SESSION_V6_DYNAMIC_ARTIFACTS = (
    (re.compile(r"\Aplanning\.assignment:ASG-[0-9a-f]{64}\Z"), "planning"),
    (re.compile(r"\Areviewer\.result:ASG-[0-9a-f]{64}\Z"), "reviewers"),
)


def session_v6_artifact_phase(logical_name: str) -> str:
    """Return the sole v6 phase allowed to own a logical artifact."""

    if type(logical_name) is not str:
        raise ValueError("Session v6 artifact name must be text")
    owner = SESSION_V6_ARTIFACT_PHASES.get(logical_name)
    if owner is not None:
        return owner
    for pattern, dynamic_owner in _SESSION_V6_DYNAMIC_ARTIFACTS:
        if pattern.fullmatch(logical_name) is not None:
            return dynamic_owner
    raise ValueError(
        f"No Session v6 artifact owner is defined for: {logical_name}"
    )

PER_REVIEWER_SCHEMAS = {
    "_envelope": "model_request_envelope_v1",
    "_raw_response": "model_raw_response_v1",
    "_result": "reviewer_result_v1",
    "_agent_trace": "reviewer_agent_trace_v1",
    "_observations": "observation_log_jsonl_v1",
}

# Dynamic artifact identifiers are deliberately narrower than filesystem-safe
# names in general.  Stable Runtime IDs use prefixed hexadecimal digests; the
# numeric alternatives retain deterministic batch/index naming used by early
# v4 writers without admitting arbitrary slugs.  In particular, none of these
# expressions can consume a slash, backslash, or dot.
_BATCH_ID = r"(?:0|[1-9][0-9]*|B-[0-9a-f]{1,64})"
_WAVE_ID = r"(?:[1-9][0-9]*|W-[0-9a-f]{64})"
_TASK_ID = r"(?:[1-9][0-9]*|STASK-[0-9a-f]{64})"

_PER_REVIEWER_PATTERN = re.compile(
    r"\Areviewer_(?:0|[1-9][0-9]*)_"
    r"(?P<kind>envelope|raw_response|result|agent_trace|observations)\Z"
)
_INTENT_DECISION_PATTERN = re.compile(
    r"\Aintent_decision_decision_[0-9a-f]{16}\Z"
)
_RECONCILER_PATTERN = re.compile(
    rf"\Areconciler_{_BATCH_ID}_"
    r"(?P<kind>envelope|raw_response|decision)\Z"
)
_SUPPLEMENTAL_WAVE_PATTERN = re.compile(
    rf"\Asupplemental_wave_{_WAVE_ID}_"
    r"(?P<kind>plan|budget|reconciler_decision|summary)\Z"
)
_SUPPLEMENTAL_TASK_PATTERN = re.compile(
    rf"\Asupplemental_task_{_TASK_ID}_"
    r"(?P<kind>spec|assignment|envelope|raw_response|result|agent_trace|observations)\Z"
)

_RECONCILER_SCHEMAS = {
    "envelope": "semantic_reconciler_envelope_v1",
    "raw_response": "semantic_reconciler_raw_response_v1",
    "decision": "semantic_reconciler_decision_v1",
}

_SUPPLEMENTAL_WAVE_SCHEMAS = {
    "plan": "supplemental_plan_v1",
    "budget": "supplemental_budget_ledger_v1",
    "reconciler_decision": "semantic_reconciler_decision_v1",
    "summary": "supplemental_wave_summary_v1",
}

_SUPPLEMENTAL_TASK_SCHEMAS = {
    "spec": "supplemental_task_spec_v1",
    "assignment": "reviewer_assignment_v1",
    "envelope": "model_request_envelope_v1",
    "raw_response": "model_raw_response_v1",
    "result": "reviewer_result_v1",
    "agent_trace": "reviewer_agent_trace_v1",
    "observations": "observation_log_jsonl_v1",
}


def artifact_schema(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError(f"No stable artifact schema is defined for: {name}")
    schema = ARTIFACT_SCHEMAS.get(name)
    if schema is not None:
        return schema
    reviewer_match = _PER_REVIEWER_PATTERN.fullmatch(name)
    if reviewer_match is not None:
        return PER_REVIEWER_SCHEMAS[f"_{reviewer_match.group('kind')}"]
    if _INTENT_DECISION_PATTERN.fullmatch(name) is not None:
        return "intent_decision_v1"
    reconciler_match = _RECONCILER_PATTERN.fullmatch(name)
    if reconciler_match is not None:
        return _RECONCILER_SCHEMAS[reconciler_match.group("kind")]
    wave_match = _SUPPLEMENTAL_WAVE_PATTERN.fullmatch(name)
    if wave_match is not None:
        return _SUPPLEMENTAL_WAVE_SCHEMAS[wave_match.group("kind")]
    task_match = _SUPPLEMENTAL_TASK_PATTERN.fullmatch(name)
    if task_match is not None:
        return _SUPPLEMENTAL_TASK_SCHEMAS[task_match.group("kind")]
    raise ValueError(f"No stable artifact schema is defined for: {name}")
