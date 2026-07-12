import pytest

from review_agent.artifacts import artifact_schema


def test_artifact_schema_resolves_stage_and_per_reviewer_artifacts() -> None:
    assert artifact_schema("request") == "review_request_v1"
    assert artifact_schema("incremental_priority") == "incremental_priority_map_v1"
    assert artifact_schema("repository_observations") == "observation_log_jsonl_v1"
    assert artifact_schema("reviewer_12_result") == "reviewer_result_v1"
    assert artifact_schema("reviewer_0_observations") == "observation_log_jsonl_v1"


def test_artifact_schema_rejects_unversioned_artifact() -> None:
    with pytest.raises(ValueError, match="stable artifact schema"):
        artifact_schema("reviewer_x_result")
