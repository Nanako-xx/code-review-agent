import pytest

from review_agent.artifacts import artifact_schema


def test_artifact_schema_resolves_stage_and_per_reviewer_artifacts() -> None:
    assert artifact_schema("request") == "review_request_v1"
    assert artifact_schema("incremental_priority") == "incremental_priority_map_v1"
    assert artifact_schema("repository_observations") == "observation_log_jsonl_v1"
    assert artifact_schema("risk_model_decision") == "risk_model_decision_v1"
    assert artifact_schema("risk_model_envelope") == "risk_model_envelope_v1"
    assert (
        artifact_schema("portfolio_model_raw_response")
        == "portfolio_model_raw_response_v1"
    )
    assert artifact_schema("portfolio_plan") == "portfolio_plan_v1"
    assert artifact_schema("planning_summary") == "planning_summary_v1"
    assert artifact_schema("reviewer_12_result") == "reviewer_result_v1"
    assert artifact_schema("reviewer_0_observations") == "observation_log_jsonl_v1"


def test_artifact_schema_resolves_semantic_reconciliation_artifacts() -> None:
    assert artifact_schema("reconciliation_prepass") == "reconciliation_prepass_v1"
    assert artifact_schema("reconciliation_packet") == "reconciliation_packet_v1"
    assert artifact_schema("supplemental_initial_plan") == "supplemental_plan_v1"
    assert (
        artifact_schema("reconciliation_analysis_summary")
        == "reconciliation_analysis_summary_v1"
    )
    assert artifact_schema("semantic_reconciliation") == "semantic_reconciliation_v1"
    assert artifact_schema("reconciliation") == "evidence_reconciliation_v1"
    assert artifact_schema("supplemental_summary") == "supplemental_summary_v1"


def test_artifact_schema_resolves_strict_batch_wave_and_task_artifacts() -> None:
    batch_id = "B-" + "a" * 32
    wave_id = "W-" + "b" * 64
    task_id = "STASK-" + "c" * 64

    assert (
        artifact_schema(f"reconciler_{batch_id}_envelope")
        == "semantic_reconciler_envelope_v1"
    )
    assert (
        artifact_schema(f"reconciler_{batch_id}_raw_response")
        == "semantic_reconciler_raw_response_v1"
    )
    assert (
        artifact_schema(f"reconciler_{batch_id}_decision")
        == "semantic_reconciler_decision_v1"
    )
    assert (
        artifact_schema(f"supplemental_wave_{wave_id}_plan")
        == "supplemental_plan_v1"
    )
    assert (
        artifact_schema(f"supplemental_wave_{wave_id}_budget")
        == "supplemental_budget_ledger_v1"
    )
    assert (
        artifact_schema(f"supplemental_wave_{wave_id}_reconciler_decision")
        == "semantic_reconciler_decision_v1"
    )
    assert (
        artifact_schema(f"supplemental_wave_{wave_id}_summary")
        == "supplemental_wave_summary_v1"
    )

    expected_task_schemas = {
        "spec": "supplemental_task_spec_v1",
        "assignment": "reviewer_assignment_v1",
        "envelope": "model_request_envelope_v1",
        "raw_response": "model_raw_response_v1",
        "result": "reviewer_result_v1",
        "agent_trace": "reviewer_agent_trace_v1",
        "observations": "observation_log_jsonl_v1",
    }
    for suffix, schema in expected_task_schemas.items():
        assert artifact_schema(f"supplemental_task_{task_id}_{suffix}") == schema


@pytest.mark.parametrize(
    "name",
    [
        "reviewer_x_result",
        "reviewer_01_result",
        "reviewer_1_result.json",
        "intent_decision_../escape",
        "reconciler_B-a/decision_envelope",
        "reconciler_B-xyz_decision",
        "reconciler_B-a_decision/extra",
        "supplemental_wave_W-../plan",
        "supplemental_wave_W-aaaaaaaa_plan",
        "supplemental_task_STASK-../result",
        "supplemental_task_STASK-aaaaaaaa_result",
        "supplemental_task_STASK-" + "a" * 64 + "_result.json",
        "../semantic_reconciliation",
    ],
)
def test_artifact_schema_rejects_unknown_traversal_and_malformed_names(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="stable artifact schema"):
        artifact_schema(name)
