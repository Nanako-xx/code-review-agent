"""Compatibility checks backed by hand-authored, immutable Session fixtures.

These fixtures intentionally do not call the current v5 serializer to manufacture
historical payloads.  Each JSON file records the exact field and phase layout that
its schema version accepted.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

import review_agent.memory_curator as memory_curator_module
import review_agent.memory_store as memory_store_module
import review_agent.legacy_resume as resume_module
from review_agent.legacy_resume import ResumeAction, ReviewSessionResumer
from review_agent.run_state import RunPhase
from review_agent.session import ModelStageConfig, session_manifest_to_dict
from review_agent.session_store import SessionStore


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "sessions"

HISTORICAL_PHASE_LAYOUTS = {
    1: (
        "preflight",
        "repository_intelligence",
        "reviewers",
        "reconciliation",
        "completion",
        "final_risk",
        "reporting",
    ),
    2: (
        "preflight",
        "quality_gates",
        "repository_intelligence",
        "intent_discovery",
        "intent_resolution",
        "planning",
        "reviewers",
        "reconciliation",
        "completion",
        "final_risk",
        "reporting",
    ),
    3: (
        "preflight",
        "quality_gates",
        "repository_intelligence",
        "intent_discovery",
        "intent_resolution",
        "planning",
        "reviewers",
        "reconciliation",
        "completion",
        "final_risk",
        "reporting",
    ),
    4: (
        "preflight",
        "quality_gates",
        "repository_intelligence",
        "intent_discovery",
        "intent_resolution",
        "planning",
        "reviewers",
        "reconciliation_analysis",
        "supplemental_investigation",
        "reconciliation",
        "completion",
        "final_risk",
        "reporting",
    ),
}

COMMON_ROOT_FIELDS = {
    "schema_version",
    "review_id",
    "parent_review_id",
    "root_review_id",
    "repository",
    "revisions",
    "execution",
    "status",
    "current_phase",
    "last_successful_phase",
    "phases",
    "artifacts",
    "errors",
    "created_at",
    "updated_at",
}

COMMON_EXECUTION_FIELDS = {
    "reviewer_provider",
    "reviewer_model",
    "reviewer_base_url",
    "reviewer_api_key_env",
    "reviewer_mode",
    "reviewer_loop",
    "non_interactive",
}

MODEL_STAGE_FIELDS = {
    "mode",
    "provider",
    "model",
    "base_url",
    "api_key_env",
    "max_output_tokens",
    "max_provider_attempts",
    "max_elapsed_seconds",
}

SUPPLEMENTAL_POLICY_FIELDS = {
    "version",
    "risk_level",
    "max_waves",
    "max_tasks",
    "max_tasks_per_wave",
    "max_concurrency",
    "max_turns_per_task",
    "max_tool_calls_per_task",
    "max_tokens_per_task",
    "max_total_tokens",
    "max_elapsed_seconds",
}

PHASE_FIELDS_V1 = {
    "status",
    "attempts",
    "started_at",
    "completed_at",
    "artifacts",
    "error",
    "tasks",
}


def _fixture_path(schema_version: int) -> Path:
    return FIXTURE_ROOT / f"v{schema_version}" / "session.json"


def _fixture_payload(schema_version: int) -> dict[str, object]:
    payload = json.loads(_fixture_path(schema_version).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _copy_fixture(tmp_path: Path, schema_version: int) -> SessionStore:
    run_dir = tmp_path / f"session-v{schema_version}"
    run_dir.mkdir()
    shutil.copyfile(_fixture_path(schema_version), run_dir / "session.json")
    return SessionStore(run_dir)


def _expected_execution_fields(schema_version: int) -> set[str]:
    fields = set(COMMON_EXECUTION_FIELDS)
    if schema_version >= 3:
        fields.update({"risk_assessor", "portfolio_planner"})
    if schema_version >= 4:
        fields.update({"semantic_reconciler", "supplemental_policy"})
    return fields


@pytest.mark.parametrize("schema_version", (1, 2, 3, 4))
def test_frozen_session_fixture_is_canonical_for_historical_schema(
    schema_version: int,
) -> None:
    raw = _fixture_payload(schema_version)
    expected_root_fields = set(COMMON_ROOT_FIELDS)
    if schema_version >= 4:
        expected_root_fields.add("supplemental_waves")

    assert raw["schema_version"] == schema_version
    assert set(raw) == expected_root_fields
    assert tuple(raw["phases"]) == HISTORICAL_PHASE_LAYOUTS[schema_version]

    execution = raw["execution"]
    assert isinstance(execution, dict)
    assert set(execution) == _expected_execution_fields(schema_version)
    assert "memory" not in execution
    assert "memory_curator" not in execution
    if schema_version >= 3:
        assert set(execution["risk_assessor"]) == MODEL_STAGE_FIELDS
        assert set(execution["portfolio_planner"]) == MODEL_STAGE_FIELDS
    if schema_version >= 4:
        assert set(execution["semantic_reconciler"]) == MODEL_STAGE_FIELDS
        assert set(execution["supplemental_policy"]) == SUPPLEMENTAL_POLICY_FIELDS

    expected_phase_fields = set(PHASE_FIELDS_V1)
    if schema_version >= 2:
        expected_phase_fields.add("user_decisions")
    for checkpoint in raw["phases"].values():
        assert set(checkpoint) == expected_phase_fields

    manifest = SessionStore(_fixture_path(schema_version).parent).load()
    assert manifest.schema_version == schema_version
    assert tuple(manifest.phases) == HISTORICAL_PHASE_LAYOUTS[schema_version]
    assert manifest.execution.memory is None
    assert manifest.execution.memory_curator == ModelStageConfig()
    assert RunPhase.MEMORY_SELECTION.value not in manifest.phases
    assert RunPhase.MEMORY_PROPOSAL.value not in manifest.phases
    assert session_manifest_to_dict(manifest) == raw


def test_frozen_v1_session_is_audit_only_and_byte_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _copy_fixture(tmp_path, 1)
    before = store.session_path.read_bytes()
    manifest = store.load()

    with pytest.raises(ValueError, match="read-only audit"):
        store.write(manifest)
    assert store.session_path.read_bytes() == before

    def forbidden_pipeline(*_args, **_kwargs):
        raise AssertionError("schema v1 audit must not construct ReviewPipeline")

    monkeypatch.setattr(resume_module, "ReviewPipeline", forbidden_pipeline)
    resumer = ReviewSessionResumer(
        repository=tmp_path,
        checkpoint_store=object(),
        session_store=store,
    )
    monkeypatch.setattr(
        resumer,
        "_validate_repository_and_revisions",
        lambda frozen: (frozen.repository, frozen.revisions),
    )
    monkeypatch.setattr(resumer, "_load_request", lambda _frozen: object())

    result = resumer.resume()

    assert result.action is ResumeAction.AUDIT_COMPLETED
    assert result.starting_phase is None
    assert tuple(phase.value for phase in result.reused_phases) == (
        HISTORICAL_PHASE_LAYOUTS[1]
    )
    assert store.session_path.read_bytes() == before


@pytest.mark.parametrize("schema_version", (2, 3, 4))
def test_frozen_resumable_session_advances_without_activating_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_version: int,
) -> None:
    store = _copy_fixture(tmp_path, schema_version)
    original = store.load()

    def forbidden_memory(*_args, **_kwargs):
        raise AssertionError("legacy Session transition must not activate Memory")

    monkeypatch.setattr(
        memory_store_module.MemoryStore,
        "__init__",
        forbidden_memory,
    )
    monkeypatch.setattr(
        memory_curator_module,
        "run_local_memory_curator",
        forbidden_memory,
    )
    monkeypatch.setattr(
        memory_curator_module,
        "run_model_memory_curator",
        forbidden_memory,
    )

    updated = store.mark_phase_running(
        RunPhase.PREFLIGHT,
        f"2026-0{schema_version}-01T00:00:01Z",
    )
    reloaded = store.load()
    persisted = json.loads(store.session_path.read_text(encoding="utf-8"))

    assert updated == reloaded
    assert updated.schema_version == schema_version
    assert updated.execution == original.execution
    assert updated.execution.memory is None
    assert updated.execution.memory_curator == ModelStageConfig()
    assert updated.current_phase is RunPhase.PREFLIGHT
    assert updated.phases[RunPhase.PREFLIGHT.value].attempts == 1
    assert tuple(updated.phases) == HISTORICAL_PHASE_LAYOUTS[schema_version]
    assert RunPhase.MEMORY_SELECTION.value not in updated.phases
    assert RunPhase.MEMORY_PROPOSAL.value not in updated.phases
    assert set(persisted["execution"]) == _expected_execution_fields(schema_version)
    assert "memory" not in persisted["execution"]
    assert "memory_curator" not in persisted["execution"]
