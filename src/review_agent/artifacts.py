from __future__ import annotations


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
    "multi_reviewer": "multi_reviewer_result_v1",
    "reviewer_envelope": "model_request_envelope_v1",
    "reviewer_raw_response": "model_raw_response_v1",
    "reviewer": "reviewer_result_v1",
    "reviewer_agent_trace": "reviewer_agent_trace_v1",
    "reconciliation": "evidence_reconciliation_v1",
    "completion": "completion_check_v1",
    "final_risk": "final_risk_assessment_v1",
    "review_brief": "review_brief_v1",
    "report": "review_report_markdown_v1",
    "observations": "observation_log_jsonl_v1",
}

PER_REVIEWER_SCHEMAS = {
    "_envelope": "model_request_envelope_v1",
    "_raw_response": "model_raw_response_v1",
    "_result": "reviewer_result_v1",
    "_agent_trace": "reviewer_agent_trace_v1",
    "_observations": "observation_log_jsonl_v1",
}


def artifact_schema(name: str) -> str:
    schema = ARTIFACT_SCHEMAS.get(name)
    if schema is not None:
        return schema
    if name.startswith("reviewer_"):
        for suffix, reviewer_schema in PER_REVIEWER_SCHEMAS.items():
            reviewer_number = name[len("reviewer_") : -len(suffix)]
            if name.endswith(suffix) and reviewer_number.isdigit():
                return reviewer_schema
    if name.startswith("intent_decision_"):
        event_id = name.removeprefix("intent_decision_")
        if event_id:
            return "intent_decision_v1"
    raise ValueError(f"No stable artifact schema is defined for: {name}")
